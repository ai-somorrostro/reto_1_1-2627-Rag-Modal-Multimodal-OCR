"""PDF a Markdown por página con Docling, render de páginas a PNG y detección de elementos repetidos (logo)."""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import CFG

IMAGE_MARKER = "<!-- image -->"


def _silence_logs() -> None:
    for name in ("docling", "docling_core"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # RapidOCR (el OCR de Docling) fija su nivel al importarse; desactivar el logger sí persiste
    logging.getLogger("RapidOCR").disabled = True


def parse(path: Path | None = None, ocr: bool = True):
    """Convierte el PDF con Docling y cachea el resultado en JSON (se reutiliza si el PDF no cambia)."""
    from docling_core.types.doc import DoclingDocument

    path = Path(path or CFG.DOC_PATH)
    cache = CFG.DOC_CACHE / "docling.json"
    if cache.exists():
        return DoclingDocument.load_from_json(cache)
    _silence_logs()
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from .models import get_device, release

    pdf_options = PdfPipelineOptions(do_ocr=ocr, generate_picture_images=True, images_scale=1.0)
    pdf_options.accelerator_options = AcceleratorOptions(device=get_device())
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)})
    doc = converter.convert(path).document
    doc.save_as_json(cache)
    del converter
    release()
    return doc


def num_pages(doc) -> int:
    return len(doc.pages)


def markdown_by_page(doc) -> dict[int, str]:
    """Markdown de cada página (solo la capa de contenido; Docling aparta cabeceras y pies)."""
    return {n: doc.export_to_markdown(page_no=n) for n in sorted(doc.pages)}


def render_pages(path: Path | None = None, dpi: int = 110) -> dict[int, Path]:
    """Guarda cada página como PNG (caché) y devuelve {página: ruta}."""
    import pymupdf

    path = Path(path or CFG.DOC_PATH)
    folder = CFG.DOC_CACHE / f"paginas_{dpi}dpi"
    folder.mkdir(exist_ok=True)
    output = {}
    with pymupdf.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            target = folder / f"pagina_{i:02d}.png"
            if not target.exists():
                page.get_pixmap(dpi=dpi).save(target)
            output[i] = target
    return output


def raw_text_by_page(path: Path | None = None) -> dict[int, str]:
    """Texto plano extraído con PyMuPDF, sin análisis de maquetación (para comparar con Docling)."""
    import pymupdf

    with pymupdf.open(Path(path or CFG.DOC_PATH)) as pdf:
        return {i: page.get_text() for i, page in enumerate(pdf, start=1)}


def pdf_images(path: Path | None = None) -> list[dict]:
    """Imágenes incrustadas en el PDF (PyMuPDF): página, identificador interno (xref) y rectángulo."""
    import pymupdf

    rows = []
    with pymupdf.open(Path(path or CFG.DOC_PATH)) as pdf:
        for n, page in enumerate(pdf, start=1):
            for info in page.get_images(full=True):
                for rect in page.get_image_rects(info[0]):
                    rows.append({"pagina": n, "xref": info[0], "ancho_px": info[2], "alto_px": info[3],
                                  "rect": tuple(round(v, 1) for v in rect)})
    return rows


def repeated_images(path: Path | None = None, threshold: float = 0.5) -> list[dict]:
    """Imágenes que se repiten en al menos `threshold` de las páginas (logo de cabecera, sellos...)."""
    import pandas as pd

    df = pd.DataFrame(pdf_images(path))
    if df.empty:
        return []
    import pymupdf

    with pymupdf.open(Path(path or CFG.DOC_PATH)) as pdf:
        n_pages = len(pdf)
    by_xref = df.groupby("xref")["pagina"].nunique()
    repeated = by_xref[by_xref >= max(2, threshold * n_pages)].index
    return df[df["xref"].isin(repeated)].to_dict("records")


def _overlap(box: tuple, rect: tuple) -> float:
    """Fracción de `box` que cae dentro de `rect` (ambas como x0, y0, x1, y1)."""
    width = max(0.0, min(box[2], rect[2]) - max(box[0], rect[0]))
    height = max(0.0, min(box[3], rect[3]) - max(box[1], rect[1]))
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return width * height / area


def docling_images(doc) -> list[dict]:
    """Imágenes que Docling ha detectado en el cuerpo, con página y caja en coordenadas de arriba-izquierda."""
    rows = []
    for pic in doc.pictures:
        if not pic.prov:
            continue
        prov = pic.prov[0]
        box = prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height)
        img = pic.get_image(doc)
        rows.append({"ref": pic.self_ref, "pagina": prov.page_no, "caja": (box.l, box.t, box.r, box.b),
                      "tamano_px": img.size if img else None, "item": pic})
    return rows


def deduplicate(doc, path: Path | None = None, threshold: float = 0.5):
    """Copia del documento sin las imágenes que coinciden con una imagen repetida del PDF (el logo).

    Devuelve (documento limpio, lista de imágenes quitadas).
    """
    repeated = repeated_images(path, threshold)
    rects = defaultdict(list)
    for r in repeated:
        rects[r["pagina"]].append(r["rect"])
    clean_doc = doc.model_copy(deep=True)
    to_remove = [f for f in docling_images(clean_doc)
              if any(_overlap(f["caja"], rect) > 0.5 for rect in rects.get(f["pagina"], []))]
    if to_remove:
        clean_doc.delete_items(node_items=[f["item"] for f in to_remove])
    return clean_doc, [{k: v for k, v in f.items() if k != "item"} for f in to_remove]


def repeated_lines(pages: dict[int, str], threshold: float = 0.5) -> set[str]:
    """Líneas de texto que se repiten en al menos `threshold` de las páginas (cabeceras, pies)."""
    counts = Counter()
    for text in pages.values():
        counts.update({l.strip() for l in text.splitlines() if l.strip() and l.strip() != IMAGE_MARKER})
    minimum = max(2, int(threshold * len(pages)))
    return {l for l, n in counts.items() if n >= minimum}


def clean_markdown(pages: dict[int, str], repeated: set[str] | None = None) -> dict[int, str]:
    """Quita líneas repetidas entre páginas y saltos de línea sobrantes."""
    repeated = repeated if repeated is not None else repeated_lines(pages)
    output = {}
    for n, text in pages.items():
        lines = [l for l in text.splitlines() if l.strip() not in repeated]
        output[n] = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return output


def clean_pages(path: Path | None = None) -> dict[int, str]:
    """Atajo del pipeline: parsear, quitar el logo y las líneas repetidas, y devolver Markdown por página."""
    doc = parse(path)
    clean_doc, _ = deduplicate(doc)
    return clean_markdown(markdown_by_page(clean_doc))
