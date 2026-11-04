"""Carga y descarga de modelos locales (fp16 en GPU, CPU como respaldo) y caché de embeddings."""
from __future__ import annotations

import gc
import hashlib
import sqlite3
import warnings
from pathlib import Path

import numpy as np

from .config import CFG

_LOADED: dict[str, object] = {}

# Prefijos e instrucciones según la ficha de cada modelo.
# "query"/"doc" son textos que se anteponen; "query_prompt_name" usa el prompt guardado en el modelo.
PROFILES = {
    "BAAI/bge-m3": {"query": "", "doc": ""},  # la ficha indica que no necesita instrucción
    "Qwen/Qwen3-Embedding-0.6B": {"query_prompt_name": "query", "doc": ""},  # instrucción solo en la consulta
    "intfloat/multilingual-e5-large": {"query": "query: ", "doc": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "doc": "passage: "},
}


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() and not CFG.FORCE_CPU else "cpu"


def gpu_memory() -> str:
    import torch

    if not torch.cuda.is_available():
        return "sin GPU"
    used_gb = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    return f"GPU: {used_gb:.2f} GB en uso, pico {peak:.2f} GB"


def _load(key: str, factory):
    if key not in _LOADED:
        try:
            from transformers.utils import logging as tlog

            tlog.disable_progress_bar()
        except ImportError:
            pass
        _LOADED[key] = factory()
    return _LOADED[key]


def release(*keys: str) -> None:
    """Descarga los modelos indicados (o todos) y vacía la memoria de la GPU."""
    for key in list(keys or _LOADED):
        _LOADED.pop(key, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def sentence_transformer(name: str, **kwargs):
    from sentence_transformers import SentenceTransformer

    def factory():
        model = SentenceTransformer(name, device=get_device(), **kwargs)
        return model.half() if get_device() == "cuda" else model

    return _load(f"st:{name}", factory)


def cross_encoder(name: str | None = None):
    from sentence_transformers import CrossEncoder

    name = name or CFG.RERANK_MODEL

    def factory():
        model = CrossEncoder(name, device=get_device(), max_length=512)
        if get_device() == "cuda":
            model.model.half()
        return model

    return _load(f"ce:{name}", factory)


class _EmbeddingCache:
    """Caché en disco (SQLite) de vectores por modelo, rol y texto."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (modelo TEXT, rol TEXT, h TEXT, vec BLOB, PRIMARY KEY (modelo, rol, h))")

    @staticmethod
    def _h(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def read(self, model: str, role: str, texts: list[str]) -> dict[str, np.ndarray]:
        hs = {self._h(t) for t in texts}
        output = {}
        rows = self.conn.execute("SELECT h, vec FROM emb WHERE modelo=? AND rol=?", (model, role)).fetchall()
        for h, vec in rows:
            if h in hs:
                output[h] = np.frombuffer(vec, dtype=np.float32)
        return output

    def save(self, model: str, role: str, texts: list[str], vecs: np.ndarray) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?,?,?,?)",
            [(model, role, self._h(t), v.astype(np.float32).tobytes()) for t, v in zip(texts, vecs)],
        )
        self.conn.commit()


_CACHE: _EmbeddingCache | None = None


def _cache() -> _EmbeddingCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = _EmbeddingCache(CFG.CACHE_DIR / "embeddings.sqlite")
    return _CACHE


def embed(texts: list[str], role: str = "doc", model: str | None = None, batch_size: int = 16) -> np.ndarray:
    """Devuelve embeddings normalizados aplicando el prefijo correcto según `role` ("query" o "doc")."""
    model = model or CFG.EMB_MODEL
    if model not in PROFILES:
        warnings.warn(f"{model} no tiene perfil de prefijos: se codifica sin prefijo.")
    profile = PROFILES.get(model, {})
    cache = _cache()
    stored = cache.read(model, role, texts)
    pending_texts = [t for t in dict.fromkeys(texts) if cache._h(t) not in stored]
    if pending_texts:
        st = sentence_transformer(model)
        kwargs = dict(batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        if role == "query" and "query_prompt_name" in profile:
            vecs = st.encode(pending_texts, prompt_name=profile["query_prompt_name"], **kwargs)
        else:
            prefix = profile.get("query" if role == "query" else "doc", "")
            vecs = st.encode([prefix + t for t in pending_texts], **kwargs)
        cache.save(model, role, pending_texts, vecs)
        stored.update({cache._h(t): v.astype(np.float32) for t, v in zip(pending_texts, vecs)})
    return np.vstack([stored[cache._h(t)] for t in texts])
