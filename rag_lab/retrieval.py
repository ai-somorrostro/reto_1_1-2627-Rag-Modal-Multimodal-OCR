"""Fusión de rankings (RRF y combinación convexa) y reranking con cross-encoder."""
from __future__ import annotations

from collections import defaultdict

from . import models
from .config import CFG


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: suma de 1 / (k + posición) de cada lista. Ignora las puntuaciones."""
    scores = defaultdict(float)
    for ranking in rankings:
        for position, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + position)
    return sorted(scores.items(), key=lambda x: -x[1])


def _minmax(ranking: list[tuple[str, float]]) -> dict[str, float]:
    if not ranking:
        return {}
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    return {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s in ranking}


def convex(dense: list[tuple[str, float]], sparse: list[tuple[str, float]], alpha: float = 0.5) -> list[tuple[str, float]]:
    """alfa * denso + (1 - alfa) * BM25, con cada puntuación normalizada min-max. Un ausente cuenta 0."""
    d, s = _minmax(dense), _minmax(sparse)
    scores = {i: alpha * d.get(i, 0.0) + (1 - alpha) * s.get(i, 0.0) for i in set(d) | set(s)}
    return sorted(scores.items(), key=lambda x: -x[1])


def rerank_units(query: str, candidates: list[dict], top: int | None = None, model: str | None = None) -> list[dict]:
    """Ordena los candidatos con el cross-encoder (bge-reranker-v2-m3) y añade `score_rerank`."""
    if not candidates:
        return []
    ce = models.cross_encoder(model)
    scores = ce.predict([(query, c["texto"]) for c in candidates], batch_size=8, show_progress_bar=False)
    del ce
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [{**c, "score_rerank": float(p)} for c, p in ranked[: top or CFG.TOP_K]]
