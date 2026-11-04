"""Tres estrategias de chunking sobre el Markdown por página, con atribución de páginas a cada fragmento.

Se concatena el Markdown de todas las páginas y se guarda en qué posición empieza cada una.
Cada estrategia devuelve intervalos (inicio, fin) del texto, así que un fragmento sabe de qué páginas sale
aunque cruce un salto de página.
"""
from __future__ import annotations

import bisect
import re

PAGE_SEPARATOR = "\n\n"


class PagedText:
    """Markdown completo con el mapa de posiciones a páginas."""

    def __init__(self, pages: dict[int, str]):
        self.starts, self.page_numbers, parts, pos = [], [], [], 0
        for n, md in sorted(pages.items()):
            if not md.strip():
                continue
            self.starts.append(pos)
            self.page_numbers.append(n)
            parts.append(md)
            pos += len(md) + len(PAGE_SEPARATOR)
        self.text = PAGE_SEPARATOR.join(parts)

    def pages(self, start_pos: int, end_pos: int) -> list[int]:
        a = bisect.bisect_right(self.starts, start_pos) - 1
        b = bisect.bisect_right(self.starts, max(start_pos, end_pos - 1)) - 1
        return self.page_numbers[max(a, 0): b + 1]


def _fragment(text: PagedText, start_pos: int, end_pos: int, strategy: str, i: int, **extra) -> dict | None:
    content_parts = text.text[start_pos:end_pos].strip()
    if not content_parts:
        return None
    return {"id": f"{strategy}-{i:04d}", "texto": content_parts, "paginas": text.pages(start_pos, end_pos),
            "estrategia": strategy, **extra}


def _merge_small(fragments: list[dict], minimum: int = 80) -> list[dict]:
    """Une al siguiente fragmento los que no llegan a `minimum` caracteres (cabeceras sueltas, restos de corte)."""
    output, pending = [], None
    for f in fragments:
        if pending:
            f = {**f, "texto": pending["texto"] + "\n\n" + f["texto"],
                 "paginas": sorted(set(pending["paginas"]) | set(f["paginas"]))}
            pending = None
        if len(f["texto"]) < minimum:
            pending = f
        else:
            output.append(f)
    if pending and output:
        output[-1] = {**output[-1], "texto": output[-1]["texto"] + "\n\n" + pending["texto"],
                      "paginas": sorted(set(output[-1]["paginas"]) | set(pending["paginas"]))}
    elif pending:
        output.append(pending)
    return output


def _number(fragments: list, strategy: str) -> list[dict]:
    output = [f for f in fragments if f]
    for i, f in enumerate(output):
        f["id"] = f"{strategy}-{i:04d}"
    return output


# --- 1. Tamaño fijo -----------------------------------------------------------------------------

def fixed_size(pages: dict[int, str], size: int = 800, overlap: int = 100) -> list[dict]:
    """Ventanas de `size` caracteres con `overlap`, sin mirar la estructura."""
    text = PagedText(pages)
    step = size - overlap
    frags = [_fragment(text, i, i + size, "fijo", 0) for i in range(0, len(text.text), step)]
    return _number(frags, "fijo")


# --- 2. Recursivo -------------------------------------------------------------------------------

SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " "]


def _split(text: str, start_pos: int, end_pos: int, size: int, seps: list[str]) -> list[tuple[int, int]]:
    if end_pos - start_pos <= size or not seps:
        return [(start_pos, end_pos)]
    sep, rest = seps[0], seps[1:]
    cuts = [m.start() + (1 if sep.startswith("\n") else len(sep)) for m in re.finditer(re.escape(sep), text[start_pos:end_pos])]
    if not cuts:
        return _split(text, start_pos, end_pos, size, rest)
    pieces, a = [], start_pos
    for c in cuts + [end_pos - start_pos]:
        b = start_pos + c if c != end_pos - start_pos else end_pos
        if b > a:
            pieces.append((a, b))
        a = b
    # Junta piezas pequeñas hasta el tamaño y parte con el siguiente separador las que se pasan
    output, current = [], None
    for a, b in pieces:
        if b - a > size:
            if current:
                output.append(current)
                current = None
            output += _split(text, a, b, size, rest)
        elif current and b - current[0] <= size:
            current = (current[0], b)
        else:
            if current:
                output.append(current)
            current = (a, b)
    if current:
        output.append(current)
    return output


def recursive(pages: dict[int, str], size: int = 800) -> list[dict]:
    """Corta por el separador más grueso posible (sección, párrafo, línea, frase, palabra)."""
    text = PagedText(pages)
    spans = _split(text.text, 0, len(text.text), size, SEPARATORS)
    frags = [f for f in (_fragment(text, a, b, "recursivo", 0) for a, b in spans) if f]
    return _number(_merge_small(frags), "recursivo")


# --- 3. Por cabeceras de Markdown, con tablas completas -----------------------------------------

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def _level(marks: str, title: str) -> int:
    """Nivel de la cabecera. Docling suele marcar todo con '##': se usa la numeración (1., 1.2., ...) si existe."""
    numbering = re.match(r"^(\d+(?:\.\d+)*)\.?\s", title)
    return numbering.group(1).count(".") + 1 if numbering else len(marks)


def _blocks(section: str) -> list[tuple[int, int, str]]:
    """Divide una sección en bloques de tabla (líneas que empiezan por '|') y bloques de texto."""
    blocks, pos = [], 0
    for m in re.finditer(r"(?:^\|.*(?:\n|$))+", section, re.M):
        if m.start() > pos:
            blocks.append((pos, m.start(), "texto"))
        blocks.append((m.start(), m.end(), "tabla"))
        pos = m.end()
    if pos < len(section):
        blocks.append((pos, len(section), "texto"))
    return blocks


def by_headers(pages: dict[int, str], max_size: int = 1500) -> list[dict]:
    """Un fragmento por sección. Las tablas nunca se parten; el texto largo se divide por párrafos."""
    text = PagedText(pages)
    headers = list(HEADER_RE.finditer(text.text))
    bounds = [0] + [m.start() for m in headers] + [len(text.text)]
    path: list[tuple[int, str]] = []
    frags = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = HEADER_RE.match(text.text, a)
        if m:
            level = _level(m.group(1), m.group(2).strip())
            path = [r for r in path if r[0] < level] + [(level, m.group(2).strip())]
        section_path = " > ".join(t for _, t in path)
        body_start = m.end() if m else a
        title = m.group(0) + "\n\n" if m else ""
        for begin, end_pos, kind in _blocks(text.text[body_start:b]):
            begin, end_pos = body_start + begin, body_start + end_pos
            if not text.text[begin:end_pos].strip():
                continue
            spans = [(begin, end_pos)] if kind == "tabla" else _split(text.text, begin, end_pos, max_size, SEPARATORS[2:])
            for x, y in spans:
                f = _fragment(text, x, y, "cabeceras", 0, seccion=section_path, tipo=kind)
                if f:
                    f["texto"] = (title + f["texto"]).strip()
                    f["paginas"] = sorted(set(f["paginas"]) | set(text.pages(a, a + 1) if m else []))
                    frags.append(f)
    # Junta bloques de texto contiguos y pequeños de la misma sección
    merged = []
    for f in frags:
        prev = merged[-1] if merged else None
        if (prev and prev["tipo"] == f["tipo"] == "texto" and prev["seccion"] == f["seccion"]
                and len(prev["texto"]) + len(f["texto"]) <= max_size):
            body = f["texto"].split("\n\n", 1)[-1] if f["texto"].startswith("#") else f["texto"]
            prev["texto"] += "\n\n" + body
            prev["paginas"] = sorted(set(prev["paginas"]) | set(f["paginas"]))
        else:
            merged.append(f)
    return _number(_merge_small(merged), "cabeceras")


STRATEGIES = {"fijo": fixed_size, "recursivo": recursive, "cabeceras": by_headers}


def summary(fragments: list[dict]) -> dict:
    import numpy as np

    lengths = [len(f["texto"]) for f in fragments]
    return {"fragmentos": len(fragments), "media_chars": int(np.mean(lengths)), "min_chars": min(lengths),
            "max_chars": max(lengths), "multipagina": sum(len(f["paginas"]) > 1 for f in fragments)}
