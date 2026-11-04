"""Búsqueda híbrida en una base de datos vectorial como servidor: Qdrant en Docker.

Cada punto guarda el vector denso (BGE-M3), el vector disperso con los pesos BM25 de bm25s y un payload
con las páginas, la sección y el tipo del fragmento. La fusión RRF la hace el propio Qdrant.
Documentación de consultas híbridas: https://qdrant.tech/documentation/concepts/hybrid-queries/
"""
from __future__ import annotations

from . import models
from .config import CFG
from .indexes import BM25Index


def qdrant_available(url: str | None = None, timeout: float = 3.0) -> bool:
    """Comprueba si Qdrant responde en /healthz. Sin respuesta, devuelve False."""
    import requests

    try:
        return requests.get((url or CFG.QDRANT_URL).rstrip("/") + "/healthz", timeout=timeout).ok
    except requests.RequestException:
        return False


class QdrantHybrid:
    """Colección de Qdrant con un vector denso y uno disperso por fragmento."""

    def __init__(self, units: list[dict], name: str, url: str | None = None, model: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm

        self.qm = qm
        self.client = QdrantClient(url=url or CFG.QDRANT_URL)
        self.name = name
        self.model = model or CFG.EMB_MODEL
        self.bm25 = BM25Index(units)
        dense = models.embed([u["texto"] for u in units], "doc", self.model)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name,
            vectors_config={"dense": qm.VectorParams(size=dense.shape[1], distance=qm.Distance.COSINE)},
            sparse_vectors_config={"bm25": qm.SparseVectorParams()},
        )
        points = []
        for i, (unit, vector, (indices, values)) in enumerate(zip(units, dense, self.bm25.doc_sparse_vectors())):
            payload = {"id": unit["id"], "paginas": unit.get("paginas", []), "seccion": unit.get("seccion", ""),
                       "tipo": unit.get("tipo", "")}
            points.append(qm.PointStruct(id=i, payload=payload, vector={
                "dense": vector.tolist(), "bm25": qm.SparseVector(indices=indices, values=values)}))
        self.client.upsert(name, points=points, wait=True)

    def search(self, query: str, k: int = 10, rrf_k: int | None = 60, kind: str | None = None,
               candidates: int = 50) -> list[tuple[str, float]]:
        """Híbrido en el servidor: 50 candidatos densos y 50 de BM25, fusionados con RRF.

        `rrf_k=None` usa el valor por defecto de Qdrant (k=2). `kind` filtra por el tipo del payload.
        """
        qm = self.qm
        flt = qm.Filter(must=[qm.FieldCondition(key="tipo", match=qm.MatchValue(value=kind))]) if kind else None
        dense = models.embed([query], "query", self.model)[0].tolist()
        indices, values = self.bm25.query_sparse(query)
        prefetch = [qm.Prefetch(query=dense, using="dense", limit=candidates, filter=flt)]
        if indices:
            prefetch.append(qm.Prefetch(query=qm.SparseVector(indices=indices, values=values), using="bm25",
                                        limit=candidates, filter=flt))
        rrf = qm.Rrf(k=rrf_k) if rrf_k is not None else qm.Rrf()
        result = self.client.query_points(self.name, prefetch=prefetch, query=qm.RrfQuery(rrf=rrf), limit=k,
                                          with_payload=True)
        return [(p.payload["id"], p.score) for p in result.points]

    def count(self) -> int:
        return self.client.count(self.name).count
