"""Índices de recuperación: denso en Chroma persistente y BM25 en español con bm25s."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from . import models
from .config import CFG


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:50]


def _signature(units: list[dict]) -> str:
    h = hashlib.sha1()
    for u in units:
        h.update(u["id"].encode())
        h.update(u["texto"].encode())
    return h.hexdigest()[:16]


class DenseIndex:
    """Colección de Chroma con embeddings calculados por nosotros (con el prefijo correcto del modelo)."""

    def __init__(self, name: str, units: list[dict], model: str | None = None, path: Path | None = None):
        import chromadb

        self.model = model or CFG.EMB_MODEL
        self.units = {u["id"]: u for u in units}
        path = path or CFG.DOC_CACHE / "chroma"
        client = chromadb.PersistentClient(path=str(path))
        collection_name = f"{_slug(name)}--{_slug(self.model)}"
        signature = _signature(units)
        self.col = client.get_or_create_collection(collection_name, configuration={"hnsw": {"space": "cosine"}},
                                                    metadata={"firma": signature}, embedding_function=None)
        if (self.col.metadata or {}).get("firma") != signature or self.col.count() != len(units):
            client.delete_collection(collection_name)
            self.col = client.create_collection(collection_name, configuration={"hnsw": {"space": "cosine"}},
                                                 metadata={"firma": signature}, embedding_function=None)
            vectors = models.embed([u["texto"] for u in units], "doc", self.model)
            for i in range(0, len(units), 500):
                unit_batch = units[i:i + 500]
                self.col.add(ids=[u["id"] for u in unit_batch], embeddings=vectors[i:i + 500].tolist(),
                             metadatas=[{"paginas": ",".join(map(str, u.get("paginas", [])))} for u in unit_batch])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        vector = models.embed([query], "query", self.model)
        return self.search_vector(vector[0], k)

    def search_vector(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        r = self.col.query(query_embeddings=[vector.tolist()], n_results=min(k, len(self.units)), include=["distances"])
        return [(i, 1.0 - d) for i, d in zip(r["ids"][0], r["distances"][0])]  # similitud coseno


class BM25Index:
    """BM25 con stemmer y stopwords en español (bm25s + PyStemmer)."""

    def __init__(self, units: list[dict], k1: float = 1.5, b: float = 0.75):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("spanish")
        self.ids = [u["id"] for u in units]
        tokens = bm25s.tokenize([u["texto"] for u in units], stopwords="es", stemmer=self.stemmer, show_progress=False)
        self.vocab = tokens.vocab  # término -> columna de la matriz de pesos BM25
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(tokens, show_progress=False)

    def doc_sparse_vectors(self) -> list[tuple[list[int], list[float]]]:
        """Pesos BM25 de cada documento como vector disperso (índices de término, valores).

        bm25s guarda la matriz documento × término ya puntuada; el producto escalar con `query_sparse`
        da la misma puntuación BM25 que `search`.
        """
        import scipy.sparse as sp

        s = self.model.scores
        matrix = sp.csc_matrix((s["data"], s["indices"], s["indptr"]),
                               shape=(s["num_docs"], len(s["indptr"]) - 1)).tocsr()
        return [(matrix[i].indices.tolist(), matrix[i].data.tolist()) for i in range(matrix.shape[0])]

    def query_sparse(self, query: str) -> tuple[list[int], list[float]]:
        """Consulta como vector disperso: cada término del vocabulario con peso igual a sus apariciones."""
        from collections import Counter

        terms = self._bm25s.tokenize([query], stopwords="es", stemmer=self.stemmer, show_progress=False,
                                     return_ids=False)[0]
        counts = Counter(self.vocab[t] for t in terms if t in self.vocab and t)
        return list(counts), [float(c) for c in counts.values()]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = self._bm25s.tokenize([query], stopwords="es", stemmer=self.stemmer, show_progress=False)
        if not tokens.vocab:
            return []
        docs, scores = self.model.retrieve(tokens, k=min(k, len(self.ids)), show_progress=False)
        return [(self.ids[d], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
