"""Corpus de control en español: SQAC (PlanTL-GOB-ES/SQAC, licencia CC BY-SA 4.0).

SQAC es un conjunto de preguntas y respuestas extractivas sobre párrafos de Wikipedia, Wikinews y AnCora.
Lo convertimos en una tarea de recuperación: el corpus son los párrafos y el documento correcto de cada
pregunta es su párrafo de origen. Se descarga con `hf_hub_download` (ficheros JSON, sin script).
"""
from __future__ import annotations

import hashlib
import json
import random

from .config import CFG

REPO = "PlanTL-GOB-ES/SQAC"


def _read(split: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, f"{split}.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def load_sqac(fast: bool | None = None) -> dict:
    """Devuelve {"docs": {id: texto}, "titulos": {id: titulo}, "consultas": [{id, texto, doc}]}.

    FAST_MODE: corpus = párrafos de dev y test (1.249) y 200 preguntas.
    Completo: se añaden los párrafos de train como distractores (6.247) y 1.000 preguntas.
    """
    fast = CFG.FAST_MODE if fast is None else fast
    query_splits = ["dev", "test"]
    corpus_splits = query_splits if fast else ["train", "dev", "test"]
    docs, titles, queries = {}, {}, []
    for split in corpus_splits:
        for article in _read(split):
            for paragraph in article["paragraphs"]:
                doc_id = "d" + hashlib.sha1(paragraph["context"].encode("utf-8")).hexdigest()[:10]
                docs[doc_id] = paragraph["context"]
                titles[doc_id] = article["title"]
                if split in query_splits:
                    queries += [{"id": qa["id"], "texto": qa["question"], "doc": doc_id} for qa in paragraph["qas"]]
    rng = random.Random(CFG.SEED)
    queries = rng.sample(queries, min(CFG.N_CONTROL_QUERIES, len(queries)))
    return {"docs": docs, "titulos": titles, "consultas": queries}
