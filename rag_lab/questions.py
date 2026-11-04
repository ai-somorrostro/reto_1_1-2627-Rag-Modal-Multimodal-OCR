"""Generación de un borrador de preguntas de evaluación para cualquier PDF.

- texto: GEN_MODEL lee el Markdown de páginas con mucho texto y sin tablas.
- tabla: GEN_MODEL lee una tabla; si hay otra casi idéntica, se le pide una pregunta que obligue a distinguirlas.
- visual: VLM_MODEL mira la imagen de las páginas que él mismo ha marcado como visuales.

El resultado es un BORRADOR: cada pregunta lleva `revisar_manual: true` y hay que revisarla a mano.
"""
from __future__ import annotations

import json
import random
import re

from . import llm, pdf
from .config import CFG

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pregunta": {"type": "string"},
        "respuesta_esperada": {"type": "string"},
    },
    "required": ["pregunta", "respuesta_esperada"],
    "additionalProperties": False,
}

VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "elemento_visual": {"type": "boolean", "description": "Hay un diagrama, cronograma, matriz, calendario, gráfico o similar"},
        "tipo": {"type": "string", "description": "diagrama, cronograma, matriz, calendario, grafico, tabla, texto u otro"},
        "descripcion": {"type": "string"},
    },
    "required": ["elemento_visual", "tipo", "descripcion"],
    "additionalProperties": False,
}

RULES = (
    "La pregunta debe estar en español, ser concreta y tener una única respuesta corta que esté en la página. "
    "Debe entenderse sin ver la página: no digas 'según la tabla' o 'en esta página'; nombra el elemento "
    "(sección, línea, tarea, objetivo, apartado). No menciones números de página. Devuelve JSON."
)


def _generate_json(prompt: str, schema: dict) -> dict:
    fmt = {"type": "json_schema", "json_schema": {"name": "pregunta", "strict": True, "schema": schema}}
    r = llm.chat([{"role": "user", "content": prompt}], CFG.GEN_MODEL, response_format=fmt,
                 temperature=0.3, max_tokens=600, step="preguntas")
    return r.json()


def _table_signature(md: str) -> set[str]:
    """Palabras de los nombres de columna, sin el prefijo de grupo ("LÍNEA 3: ... - ALUMNADO" -> "alumnado").

    Dos tablas con las mismas columnas (las de resultados de cada línea, las de cada objetivo) dan la misma firma.
    """
    header = next((l for l in md.splitlines() if l.strip().startswith("|")), "")
    cells = [c.split(" - ")[-1] for c in header.strip().strip("|").split("|")]
    return set(re.findall(r"[a-záéíóúñü]{3,}", " ".join(cells).lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def tables_by_page(doc) -> dict[int, list[str]]:
    output: dict[int, list[str]] = {}
    for t in doc.tables:
        if t.prov:
            output.setdefault(t.prov[0].page_no, []).append(t.export_to_markdown(doc))
    return output


def classify_pages() -> dict[int, dict]:
    """El VLM indica, para cada página, si tiene un elemento visual y de qué tipo."""
    images = pdf.render_pages()
    prompt = ("Mira esta página de un documento. ¿Contiene un elemento visual cuya información se entiende mejor "
              "viendo la imagen (diagrama, cronograma con celdas marcadas, matriz, calendario de colores, gráfico)? "
              "Una tabla normal de cifras no cuenta. Responde en español con JSON.")
    return {n: llm.vision(prompt, [path], schema=VISUAL_SCHEMA, max_tokens=600, step="clasificar_pagina").json()
            for n, path in images.items()}


def generate_questions(n_text: int = 7, n_table: int = 7, n_visual: int = 6, save: bool = True) -> list[dict]:
    """Genera el borrador de preguntas del PDF en CFG.DOC_PATH y lo guarda en CFG.QUESTIONS_PATH."""
    rng = random.Random(CFG.SEED)
    doc = pdf.parse()
    pages = pdf.clean_pages()
    tables = tables_by_page(doc)
    questions = []

    # Texto: páginas sin tablas, ordenadas por cantidad de texto
    options = sorted((n for n in pages if n not in tables), key=lambda n: -len(pages[n]))
    for n in (options * n_text)[:n_text]:
        asked = [p["pregunta"] for p in questions if n in p["paginas"]]
        avoid = f"\nNo repitas estas preguntas: {asked}" if asked else ""
        payload = _generate_json(f"{RULES}{avoid}\n\nPÁGINA {n}:\n{pages[n][:6000]}", QUESTION_SCHEMA)
        questions.append({**payload, "paginas": [n], "tipo": "texto"})

    # Tabla: si una tabla tiene una gemela casi idéntica en otra página, se pide distinguirlas
    table_list = [(n, md) for n, mds in sorted(tables.items()) for md in mds if md.count("\n") >= 2]

    def twins_of(n, md):
        return [(m, other) for m, other in table_list
                if other != md and _jaccard(_table_signature(md), _table_signature(other)) >= 0.7]

    # Primero tablas con gemela, luego el resto; una por página mientras haya páginas sin usar
    order = sorted(table_list, key=lambda t: (not twins_of(*t), rng.random()))
    selected, used_pages = [], set()
    for t in order + order:
        if len(selected) == n_table:
            break
        if t[0] not in used_pages or len(used_pages) == len(tables):
            selected.append(t)
            used_pages.add(t[0])
    for n, md in selected:
        twins = twins_of(n, md)
        context = f"TABLA (página {n}):\n{md[:6000]}"
        if twins:
            m, other = twins[0]
            context += (f"\n\nOJO: existe otra tabla casi idéntica (página {m}):\n{other[:3000]}\n"
                         "Escribe una pregunta cuya respuesta sea un valor que cambia entre ambas tablas, de modo que "
                         "solo la primera tabla la responda. Nombra lo que identifica a la primera tabla "
                         "(por ejemplo, la línea de actuación o el objetivo).")
        asked = [p["pregunta"] for p in questions]
        avoid = f"\nNo repitas estas preguntas: {asked}" if asked else ""
        payload = _generate_json(f"{RULES}{avoid}\nLa respuesta debe estar en la tabla.\n\n{context}", QUESTION_SCHEMA)
        questions.append({**payload, "paginas": [n], "tipo": "tabla", "tabla_gemela": twins[0][0] if twins else None})

    # Visual: el VLM genera la pregunta mirando la imagen de la página
    classification = classify_pages()
    visuals = [n for n, c in classification.items()
                if c["elemento_visual"] and c["tipo"].lower() not in ("tabla", "texto")] or list(classification)
    images = pdf.render_pages()
    for i, n in enumerate((visuals * n_visual)[:n_visual]):
        asked = [p["pregunta"] for p in questions if n in p["paginas"]]
        avoid = f"\nNo repitas estas preguntas: {asked}" if asked else ""
        prompt = (f"{RULES}{avoid}\nLa página contiene un elemento visual ({classification[n]['tipo']}). Escribe una "
                  "pregunta que solo se pueda responder mirando ese elemento (por ejemplo, qué celdas están marcadas, "
                  "qué color tiene una celda o qué conecta una flecha), no el texto corrido.")
        payload = llm.vision(prompt, [images[n]], schema=QUESTION_SCHEMA, temperature=0.3, max_tokens=600,
                        step="preguntas").json()
        questions.append({**payload, "paginas": [n], "tipo": "visual", "elemento": classification[n]["tipo"]})

    for i, p in enumerate(questions, start=1):
        p.update({"id": f"q{i:02d}", "origen": "generada por LLM", "revisar_manual": True})
    if save:
        CFG.QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CFG.QUESTIONS_PATH, "w", encoding="utf-8") as f:
            for p in questions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return questions
