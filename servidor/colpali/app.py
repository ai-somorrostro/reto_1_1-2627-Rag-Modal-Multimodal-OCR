"""Servicio opcional de recuperación visual con ColQwen/ColPali (FastAPI).

Al arrancar indexa las páginas del PDF (PDF_PATH) y expone:
  GET  /health          -> estado, modelo, páginas indexadas y memoria de GPU
  POST /search          -> {"query": str, "k": int} -> {"results": [{"page": int, "score": float}]}

El modelo se carga con sentence-transformers (MultiVectorEncoder), que reproduce el formato de consulta
con el que se entrenó el checkpoint, según su ficha en Hugging Face. Por defecto, vidore/colqwen2-v1.0
(probado); vidore/colqwen2.5-v0.2 con COLPALI_MODEL.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import MultiVectorEncoder

MODEL = os.getenv("COLPALI_MODEL", "vidore/colqwen2-v1.0")
PDF_PATH = Path(os.getenv("PDF_PATH", "documento.pdf"))
DPI = int(os.getenv("DPI", "110"))
STATE = {}


def pages_as_images(path: Path, dpi: int) -> list[Image.Image]:
    with pymupdf.open(path) as pdf:
        images = []
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    model = MultiVectorEncoder(MODEL, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    images = pages_as_images(PDF_PATH, DPI)
    with torch.no_grad():
        STATE["docs"] = model.encode_document(images, batch_size=4)
    STATE.update(model=model, n_pages=len(images))
    yield
    STATE.clear()


app = FastAPI(title="Servicio ColQwen", lifespan=lifespan_handler)


class QueryRequest(BaseModel):
    query: str
    k: int = 10


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "pages": STATE.get("n_pages", 0),
            "gpu_gb_used": round(torch.cuda.memory_allocated() / 2**30, 2),
            "gpu_gb_peak": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


@app.post("/search")
def search(c: QueryRequest):
    model = STATE["model"]
    with torch.no_grad():
        q = model.encode_query([c.query])
        scores = model.similarity(q, STATE["docs"])[0]
    order = torch.argsort(scores, descending=True)[: c.k]
    return {"results": [{"page": int(i) + 1, "score": float(scores[i])} for i in order]}
