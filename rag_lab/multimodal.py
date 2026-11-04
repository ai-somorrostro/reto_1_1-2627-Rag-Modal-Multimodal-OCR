"""RAG multimodal: descripciones de página con el VLM, CLIP multilingüe / SigLIP 2 y cliente del servicio ColPali.

Tres formas de convertir una página en algo buscable ("todo es token"):
  (a) texto: el VLM describe la página y buscamos sobre esa descripción con el RAG de texto;
  (b) un vector por imagen: CLIP o SigLIP codifican la imagen entera de la página o de la figura;
  (c) un vector por parche: ColPali/ColQwen (servicio en el servidor, no cabe en 6 GB).
"""
from __future__ import annotations

import numpy as np

from . import llm, models, pdf
from .config import CFG
from .versions import QueryTransform

PROMPT_DESCRIPTION = (
    "Describe en español el contenido de esta página de un documento para que se pueda buscar por texto. "
    "Empieza por el tipo de contenido (texto, tabla, diagrama, cronograma, matriz, calendario de colores...). "
    "Transcribe los datos que solo se ven en la imagen: en un diagrama, las cajas y flechas; en un cronograma, "
    "qué tareas están marcadas en qué meses; en una matriz, qué celdas están marcadas; en un calendario de colores, "
    "qué color tiene cada celda y qué significa según la leyenda. Incluye títulos y cifras. No inventes nada."
)


# --- (a) Descripción de páginas con el VLM -----------------------------------------------------------

def describe_pages(pages: list[int] | None = None) -> dict[int, str]:
    """Descripción de cada página generada por VLM_MODEL (cacheada como cualquier llamada al LLM)."""
    images = pdf.render_pages()
    pages = pages or sorted(images)
    return {n: llm.vision(PROMPT_DESCRIPTION, [images[n]], max_tokens=1500, step="describir_pagina").text
            for n in pages}


def visual_units(descriptions: dict[int, str]) -> list[dict]:
    return [{"id": f"pagina-{n:03d}", "texto": f"Descripción de la página {n}:\n{d}", "paginas": [n],
             "tipo": "visual", "seccion": f"Página {n}"} for n, d in sorted(descriptions.items())]


# --- (b) CLIP multilingüe y SigLIP 2 -----------------------------------------------------------------

class ImageEncoder:
    """Codifica imágenes y consultas en el mismo espacio. `family` = "clip" o "siglip"."""

    def __init__(self, family: str = "clip"):
        self.family = family

    def _siglip(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        def factory():
            model = AutoModel.from_pretrained(CFG.SIGLIP_MODEL, dtype=torch.float16 if models.get_device() == "cuda" else torch.float32)
            return model.to(models.get_device()).eval(), AutoProcessor.from_pretrained(CFG.SIGLIP_MODEL)

        return models._load(f"siglip:{CFG.SIGLIP_MODEL}", factory)

    def images(self, images: list) -> np.ndarray:
        from PIL import Image

        imgs = [Image.open(i).convert("RGB") if not hasattr(i, "convert") else i.convert("RGB") for i in images]
        if self.family == "clip":
            st = models.sentence_transformer(CFG.CLIP_IMAGE_MODEL)
            return st.encode(imgs, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        import torch

        model, proc = self._siglip()
        with torch.no_grad():
            inputs = proc(images=imgs, return_tensors="pt").to(model.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            output = model.get_image_features(**inputs)
        vec = output if torch.is_tensor(output) else output.pooler_output
        return torch.nn.functional.normalize(vec.float(), dim=-1).cpu().numpy()

    def texts(self, texts: list[str]) -> np.ndarray:
        if self.family == "clip":
            st = models.sentence_transformer(CFG.CLIP_TEXT_MODEL)
            return st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        import torch

        model, proc = self._siglip()
        with torch.no_grad():
            inputs = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                           truncation=True, return_tensors="pt").to(model.device)
            output = model.get_text_features(**inputs)
        vec = output if torch.is_tensor(output) else output.pooler_output
        return torch.nn.functional.normalize(vec.float(), dim=-1).cpu().numpy()


class ImageIndex:
    """Índice en memoria de páginas (y figuras) como imágenes. Cada unidad es {id, imagen, paginas}."""

    def __init__(self, units: list[dict], family: str = "clip"):
        self.searcher = ImageEncoder(family)
        self.units = units
        self.vectors = self.searcher.images([u["imagen"] for u in units])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.searcher.texts([query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [(self.units[i]["id"], float(scores[i])) for i in order]


def image_units(with_figures: bool = True) -> list[dict]:
    """Una unidad por página renderizada y, opcionalmente, una por figura detectada por Docling (sin el logo)."""
    units = [{"id": f"pagina-{n:03d}", "imagen": path, "paginas": [n]} for n, path in pdf.render_pages().items()]
    if with_figures:
        clean_doc, _ = pdf.deduplicate(pdf.parse())
        for i, pic in enumerate(clean_doc.pictures):
            img = pic.get_image(clean_doc)
            if img is not None and pic.prov and min(img.size) >= 32:
                units.append({"id": f"figura-{i:03d}", "imagen": img, "paginas": [pic.prov[0].page_no],
                                 "figura_de": f"pagina-{pic.prov[0].page_no:03d}"})
    return units


# --- (c) Servicio ColPali en el servidor -------------------------------------------------------------

def colpali_available(timeout: float = 3.0) -> bool:
    """Comprueba si COLPALI_URL responde en /health. Sin URL o sin respuesta, devuelve False."""
    if not CFG.COLPALI_URL:
        return False
    import requests

    try:
        return requests.get(CFG.COLPALI_URL.rstrip("/") + "/health", timeout=timeout).ok
    except requests.RequestException:
        return False


def colpali_search(query: str, k: int = 10) -> list[tuple[str, float]]:
    import requests

    r = requests.post(CFG.COLPALI_URL.rstrip("/") + "/search", json={"query": query, "k": k}, timeout=60)
    r.raise_for_status()
    return [(f"pagina-{x['page']:03d}", float(x["score"])) for x in r.json()["results"]]


# --- V7 ----------------------------------------------------------------------------------------------

class Multimodal(QueryTransform):
    """V7: V6 + unidades visuales. Las descripciones de página entran en el índice híbrido junto a los fragmentos
    de texto, y el ranking de imágenes (CLIP o SigLIP) se fusiona con RRF antes del reranker."""

    def __init__(self, text_units: list[dict], descriptions: dict[int, str], image_family: str = "clip",
                 mode: str = "multiquery"):
        visuals = visual_units(descriptions)
        super().__init__(text_units + visuals, "texto-y-visual", mode=mode, version="V7", notebook="07")
        self.technique = f"V6 + unidades visuales (descripción VLM + {image_family.upper()}) fusionadas con RRF"
        self.images = ImageIndex(image_units(with_figures=True), image_family)

    def rankings(self, question, extra=None):
        output = super().rankings(question, extra)
        # Las figuras se cuentan como su página: el reranker lee la descripción de esa página
        return output + [self._images_to_page(self.images.search(question, self.n))]

    def _images_to_page(self, ranking):
        seen, output = set(), []
        by_id = {u["id"]: u for u in self.images.units}
        for uid, s in ranking:
            page = by_id[uid].get("figura_de", uid)
            if page not in seen:
                seen.add(page)
                output.append((page, s))
        return output

