"""Métricas de recuperación escritas a mano, con verdad de referencia por página (o por documento).

Una unidad recuperada (fragmento, página, figura...) lleva el conjunto de páginas de las que sale.
Hay acierto en la posición i si alguna de sus páginas está entre las páginas correctas.
Así sirve para cualquier estrategia de chunking.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _hits(unit: Iterable, correct: set) -> bool:
    return bool(set(unit) & correct)


def recall_at_k(retrieved: Sequence[Iterable], correct: Iterable, k: int) -> float:
    """1 si alguna de las k primeras unidades proviene de una página correcta, 0 si no."""
    correct = set(correct)
    return float(any(_hits(u, correct) for u in retrieved[:k]))


def mrr_at_k(retrieved: Sequence[Iterable], correct: Iterable, k: int) -> float:
    """Inverso de la posición de la primera unidad correcta dentro de las k primeras (0 si no hay)."""
    correct = set(correct)
    for position, unit in enumerate(retrieved[:k], start=1):
        if _hits(unit, correct):
            return 1.0 / position
    return 0.0


def evaluate_retrieval(results: list[dict], ks=(5, 10)) -> dict:
    """`results`: lista de {"recuperadas": [[pags], ...], "correctas": [pags], "tipo": str}.

    Devuelve recall@k y MRR@k medios, y recall@5 por tipo de pregunta.
    """
    output = {}
    for k in ks:
        output[f"recall@{k}"] = float(np.mean([recall_at_k(r["recuperadas"], r["correctas"], k) for r in results]))
    output["MRR@10"] = float(np.mean([mrr_at_k(r["recuperadas"], r["correctas"], 10) for r in results]))
    kinds = sorted({r.get("tipo") for r in results if r.get("tipo")})
    for kind in kinds:
        of_type = [r for r in results if r.get("tipo") == kind]
        output[f"recall@5_{kind}"] = float(np.mean([recall_at_k(r["recuperadas"], r["correctas"], 5) for r in of_type]))
    return output


def latency_percentiles(latencies_ms: Sequence[float]) -> dict:
    return {
        "latencia_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latencia_p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def load_questions(path=None) -> list[dict]:
    import json

    from .config import CFG

    path = path or CFG.QUESTIONS_PATH
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate_rag(rag, questions: list[dict], corpus: str = "pdf", with_judge: bool = True, register: bool = True) -> dict:
    """Ejecuta `rag.answer` sobre las preguntas y calcula todas las columnas de results.csv.

    Latencia, tokens y coste salen de las trazas de cada pregunta y cubren solo el pipeline
    (recuperación, transformación de la consulta, reranking y generación), no el judge.
    """
    import pandas as pd

    from . import judge, results, tracing

    rows = []
    for p in questions:
        output = rag.answer(p["pregunta"])
        row = {"id": p["id"], "tipo": p.get("tipo"), "pregunta": p["pregunta"], "correctas": p["paginas"],
                "recuperadas": output["paginas_recuperadas"], "respuesta": output["respuesta"],
                "contexto": output["contexto"], "trace_id": output["trace_id"]}
        if with_judge:
            with tracing.trace(f"{rag.version}-judge", p["pregunta"]):
                verdicts = judge.judge_triad(p["pregunta"], output["respuesta"], output["contexto"])
            row.update({c: verdicts[c] for c in judge.CRITERIA})
        rows.append(row)
    detail = pd.DataFrame(rows)

    events = tracing.events_df()
    events = events[events["trace_id"].isin(detail["trace_id"])]
    by_trace = events.groupby("trace_id").agg(latencia_ms=("latencia_ms", "sum"), tokens=("tokens", "sum"),
                                                coste_usd=("coste_usd", "sum"))
    detail = detail.merge(by_trace, left_on="trace_id", right_index=True, how="left")

    metrics = evaluate_retrieval(rows)
    summary = {"version": rag.version, "notebook": rag.notebook, "tecnica": rag.technique, "corpus": corpus,
               **metrics, **latency_percentiles(detail["latencia_ms"]),
               "tokens": int(detail["tokens"].sum()), "coste_usd": float(detail["coste_usd"].sum())}
    if with_judge:
        summary.update({c: float(detail[c].mean()) for c in judge.CRITERIA})
    if register:
        results.register(summary)
    tracing.flush()
    return {"resumen": summary, "detalle": detail}
