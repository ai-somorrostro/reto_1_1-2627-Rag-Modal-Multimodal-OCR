"""Configuración única del laboratorio.

Todos los módulos leen `CFG` en el momento de usarlo, así que basta con cambiar
un atributo en la primera celda del notebook (por ejemplo `CFG.DOC_PATH = ...`).
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Salidas limpias en los notebooks: sin barras de progreso de descarga y carga de pesos
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _bool(name: str, fallback: str) -> bool:
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "si", "sí", "yes"}


@dataclass
class Config:
    # Documento y carpetas
    DOC_PATH: Path = field(default_factory=lambda: Path(os.getenv("DOC_PATH", ROOT / "data" / "documento.pdf")))
    DATA_DIR: Path = ROOT / "data"
    CACHE_DIR: Path = ROOT / "cache"
    TRACES_DIR: Path = ROOT / "trazas"
    RESULTS_CSV: Path = ROOT / "results.csv"
    QUESTIONS_PATH: Path = ROOT / "data" / "preguntas.jsonl"
    LABELS_PATH: Path = ROOT / "data" / "etiquetas_humanas.csv"

    # Ejecución
    FAST_MODE: bool = field(default_factory=lambda: _bool("FAST_MODE", "1"))
    FORCE_CPU: bool = field(default_factory=lambda: _bool("FORCE_CPU", "0"))
    SEED: int = 42

    # LLM por OpenRouter
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    GEN_MODEL: str = field(default_factory=lambda: os.getenv("GEN_MODEL", "google/gemini-3.1-flash-lite"))
    JUDGE_MODEL: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "openai/gpt-5-mini"))
    VLM_MODEL: str = field(default_factory=lambda: os.getenv("VLM_MODEL", "google/gemini-3.1-flash-lite"))

    # Modelos locales
    EMB_MODEL: str = "BAAI/bge-m3"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    CLIP_TEXT_MODEL: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    CLIP_IMAGE_MODEL: str = "sentence-transformers/clip-ViT-B-32"
    SIGLIP_MODEL: str = "google/siglip2-base-patch16-256"

    # Recuperación
    TOP_K: int = 10
    RERANK_CANDIDATES: int = 50
    RERANK_CANDIDATES_CPU: int = 10  # en CPU el reranker tarda unos 33 s por pregunta con 50 candidatos
    K_CONTEXT: int = 5  # fragmentos que se pasan al generador

    # Observabilidad y servicios
    STUDENT: str = field(default_factory=lambda: os.getenv("ALUMNO", "") or os.getenv("USER", "anonimo"))
    COLPALI_URL: str = field(default_factory=lambda: os.getenv("COLPALI_URL", ""))
    QDRANT_URL: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))

    @property
    def DOC_ID(self) -> str:
        """Identificador del PDF: nombre + hash del contenido. Separa las cachés de cada documento."""
        path = Path(self.DOC_PATH)
        h = hashlib.sha1(path.read_bytes()).hexdigest()[:8] if path.exists() else "sin_pdf"
        return f"{path.stem}_{h}"

    @property
    def DOC_CACHE(self) -> Path:
        path = self.CACHE_DIR / self.DOC_ID
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Tamaños según FAST_MODE
    @property
    def N_RERANK(self) -> int:
        """Candidatos que reordena el reranker: 50 en GPU, menos en CPU para que el notebook quepa en la clase."""
        from .models import get_device

        return self.RERANK_CANDIDATES if get_device() == "cuda" else self.RERANK_CANDIDATES_CPU

    @property
    def N_CONTROL_QUERIES(self) -> int:
        return 200 if self.FAST_MODE else 1000


CFG = Config()


def set_seeds(seed: int | None = None) -> None:
    seed = CFG.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def summary() -> None:
    """Imprime la configuración activa (sin claves)."""
    from . import models

    print(f"Documento:   {CFG.DOC_PATH} ({'existe' if Path(CFG.DOC_PATH).exists() else 'NO EXISTE'})")
    print(f"FAST_MODE:   {CFG.FAST_MODE}")
    print(f"Dispositivo: {models.get_device()}")
    print(f"GEN_MODEL:   {CFG.GEN_MODEL}")
    print(f"JUDGE_MODEL: {CFG.JUDGE_MODEL}")
    print(f"VLM_MODEL:   {CFG.VLM_MODEL}")
    print(f"OpenRouter:  {'clave configurada' if os.getenv('OPENROUTER_API_KEY') else 'SIN CLAVE (solo caché)'}")
    from . import tracing

    print(f"Trazas:      {tracing.backend()}  (alumno: {CFG.STUDENT})")
