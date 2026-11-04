"""Versiones del RAG con una interfaz común: `answer(question) -> {respuesta, contexto, paginas_recuperadas}`.

V0  chunking de tamaño fijo + denso
V1  chunking por cabeceras (tablas completas) + denso
V2  BM25 en español sobre los fragmentos de V1
V3  híbrido BM25 + denso con RRF
V4  V3 + reranker sobre el top-50
V5  V4 + enriquecimiento (ruta de sección, cabecera de contexto, tablas serializadas con su título)
V6  V5 + transformación de la consulta (multi-query o HyDE)
V7  V6 + índice de unidades visuales (ver multimodal.py)
"""
from __future__ import annotations

import re

from . import chunking, llm, pdf
from .config import CFG
from .indexes import BM25Index, DenseIndex
from .retrieval import rerank_units, rrf
from .tracing import traced, trace

GENERATOR_SYSTEM = (
    "Eres un asistente que responde preguntas sobre un documento. Responde en español, de forma breve y precisa, "
    "usando solo la información del contexto. Cita las páginas entre corchetes, por ejemplo [p. 3]. "
    "Si el contexto no contiene la respuesta, responde exactamente: 'No aparece en el documento.'"
)


def format_context(units: list[dict]) -> str:
    blocks = []
    for i, u in enumerate(units, start=1):
        pages = ", ".join(map(str, u.get("paginas", [])))
        blocks.append(f"[Fragmento {i} | páginas {pages}]\n{u.get('texto_contexto', u['texto'])}")
    return "\n\n".join(blocks)


@traced("generate")
def generate_answer(question: str, units: list[dict]) -> llm.LLMResponse:
    prompt = f"CONTEXTO:\n{format_context(units)}\n\nPREGUNTA: {question}"
    return llm.generate(prompt, system=GENERATOR_SYSTEM, max_tokens=600)


class RAG:
    """Base: subclases implementan `_retrieve(question) -> lista ordenada de unidades`."""

    version = "V?"
    technique = ""
    notebook = ""

    def __init__(self, units: list[dict]):
        self.units = {u["id"]: u for u in units}

    def _units(self, ranking: list[tuple[str, float]], k: int) -> list[dict]:
        return [{**self.units[i], "score": s} for i, s in ranking[:k] if i in self.units]

    def _retrieve(self, question: str) -> list[dict]:
        raise NotImplementedError

    def retrieve(self, question: str) -> list[dict]:
        return self._retrieve(question)[: CFG.TOP_K]

    def answer(self, question: str) -> dict:
        with trace(self.version, question) as t:
            retrieved = self.retrieve(question)
            context = retrieved[: CFG.K_CONTEXT]
            r = generate_answer(question, context)
            t["salida"] = r.text
        return {
            "respuesta": r.text,
            "contexto": format_context(context),
            "paginas_recuperadas": [u.get("paginas", []) for u in retrieved],
            "ids_recuperados": [u["id"] for u in retrieved],
            "trace_id": t["trace_id"],
            "llm": [r] + getattr(self, "_llm_extra", []),
        }


class Dense(RAG):
    def __init__(self, units, name: str, version: str, technique: str, notebook: str, model: str | None = None):
        super().__init__(units)
        self.version, self.technique, self.notebook = version, technique, notebook
        self.search_index = DenseIndex(name, units, model)

    @traced("retrieve")
    def _retrieve(self, question):
        return self._units(self.search_index.search(question, CFG.TOP_K), CFG.TOP_K)


class BM25(RAG):
    def __init__(self, units, version="V2", technique="BM25 (bm25s, stemmer y stopwords en español)", notebook="04"):
        super().__init__(units)
        self.version, self.technique, self.notebook = version, technique, notebook
        self.search_index = BM25Index(units)

    @traced("retrieve")
    def _retrieve(self, question):
        return self._units(self.search_index.search(question, CFG.TOP_K), CFG.TOP_K)


class Hybrid(RAG):
    """RRF de denso y BM25. Con `rerank=True` reordena el top-N con el cross-encoder (V4 en adelante)."""

    def __init__(self, units, name: str, version="V3", technique="Híbrido BM25 + denso con RRF", notebook="04",
                 rerank: bool = False, n_candidates: int | None = None):
        super().__init__(units)
        self.version, self.technique, self.notebook = version, technique, notebook
        self.dense = DenseIndex(name, units)
        self.bm25 = BM25Index(units)
        self.rerank = rerank
        self.n = n_candidates or (CFG.N_RERANK if rerank else CFG.TOP_K)
        if rerank and not n_candidates and CFG.N_RERANK != CFG.RERANK_CANDIDATES:
            print(f"Aviso: sin GPU el reranker reordena {CFG.N_RERANK} candidatos en lugar de {CFG.RERANK_CANDIDATES}.")

    def prepare(self, question: str):
        """Paso previo a la recuperación (V6 lo usa para transformar la consulta)."""
        return None

    def rankings(self, question: str, extra=None) -> list[list[tuple[str, float]]]:
        return [self.dense.search(question, self.n), self.bm25.search(question, self.n)]

    @traced("retrieve")
    def candidates(self, question: str, extra=None) -> list[dict]:
        return self._units(rrf(self.rankings(question, extra)), self.n)

    @traced("rerank")
    def rerank_candidates(self, question: str, candidates: list[dict]) -> list[dict]:
        return rerank_units(question, candidates, CFG.TOP_K)

    def _retrieve(self, question):
        cands = self.candidates(question, self.prepare(question))
        return self.rerank_candidates(question, cands) if self.rerank else cands


# --- V5: enriquecimiento de metadatos ---------------------------------------------------------------

def serialize_table(md: str, title: str) -> str:
    """Convierte una tabla Markdown en líneas 'columna: valor' precedidas del título de la tabla."""
    rows = [l for l in md.splitlines() if l.strip().startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", l.strip())]
    if len(rows) < 2:
        return md
    cells = [[c.strip() for c in f.strip().strip("|").split("|")] for f in rows]
    header, payload = cells[0], cells[1:]
    lines = [f"Tabla: {title}"]
    for row in payload:
        # Una columna sin nombre (la primera de una matriz) es la etiqueta de la fila
        label = next((v for c, v in zip(header, row) if not c and v), "")
        pairs = [f"{c}: {v}" for c, v in zip(header, row) if v and c]
        if pairs:
            lines.append((f"{label} | " if label else "") + "; ".join(pairs))
    return "\n".join(lines)


def enrich(fragments: list[dict], document_title: str) -> list[dict]:
    """Antepone una cabecera de contexto (documento, ruta de sección, páginas) y serializa las tablas.

    `texto` es lo que se indexa; `texto_contexto` es lo que ve el generador.
    """
    output = []
    for f in fragments:
        section = f.get("seccion") or "(sin sección)"
        header = f"Documento: {document_title} | Sección: {section} | Páginas: {', '.join(map(str, f['paginas']))}"
        body = f["texto"]
        if f.get("tipo") == "tabla":
            table_md = "\n".join(l for l in body.splitlines() if l.strip().startswith("|"))
            body = serialize_table(table_md, section.split(" > ")[-1]) + "\n\n" + table_md
        text = f"{header}\n{body}"
        output.append({**f, "texto": text, "texto_contexto": text})
    return output


def document_title(pages: dict[int, str]) -> str:
    """Título del documento: la primera cabecera de Markdown, o el nombre del fichero."""
    for n in sorted(pages):
        m = re.search(r"^#{1,6}\s+(.+)$", pages[n], re.M)
        if m:
            return m.group(1).strip()
    return CFG.DOC_PATH.stem


# --- V6: transformación de la consulta --------------------------------------------------------------

PROMPT_MULTIQUERY = (
    "Reformula la pregunta de 3 maneras distintas, en español, para buscar en un documento técnico. "
    "Usa sinónimos y términos que podrían aparecer en el documento, y conserva nombres, números y siglas. "
    "Devuelve solo las 3 reformulaciones, una por línea, sin numerar.\n\nPregunta: {question}"
)
PROMPT_HYDE = (
    "Escribe un párrafo breve, en español, que podría aparecer en la memoria de un proyecto y que responda a "
    "la pregunta. No digas que es hipotético.\n\nPregunta: {question}"
)


@traced("query_transform")
def transform_query(prompt: str) -> llm.LLMResponse:
    return llm.generate(prompt, max_tokens=300, step="query_transform")


class QueryTransform(Hybrid):
    """V6: híbrido con reranker y transformación de la consulta. `mode` = "multiquery" o "hyde"."""

    def __init__(self, units, name: str, mode: str = "multiquery", version="V6", notebook="05"):
        technique = {"multiquery": "V5 + multi-query (3 reformulaciones, RRF)",
                   "hyde": "V5 + HyDE (documento hipotético para el denso)"}[mode]
        super().__init__(units, name, version=version, technique=technique, notebook=notebook, rerank=True)
        self.mode = mode
        self._llm_extra = []

    def prepare(self, question):
        prompt = PROMPT_MULTIQUERY if self.mode == "multiquery" else PROMPT_HYDE
        r = transform_query(prompt.format(question=question))
        self._llm_extra = [r]
        if self.mode == "multiquery":
            return [l.strip(" -•\t") for l in r.text.splitlines() if l.strip()][:3]
        return r.text

    def rankings(self, question, extra=None):
        if self.mode == "multiquery":
            output = []
            for query in [question] + list(extra or []):
                output += [self.dense.search(query, self.n), self.bm25.search(query, self.n)]
            return output
        from . import models

        vector = models.embed([extra], "doc", self.dense.model)[0]  # el documento hipotético se codifica como documento
        return [self.dense.search_vector(vector, self.n), self.bm25.search(question, self.n)]


# --- Construcción de las versiones de texto ---------------------------------------------------------

def pdf_fragments(strategy: str) -> list[dict]:
    return chunking.STRATEGIES[strategy](pdf.clean_pages())


def build(version: str, v6_mode: str = "multiquery"):
    """Crea la versión pedida sobre el PDF de CFG.DOC_PATH."""
    if version == "V0":
        return Dense(pdf_fragments("fijo"), "fijo", "V0", "Chunking fijo (800/100) + denso BGE-M3", "03")
    structured = pdf_fragments("cabeceras")
    if version == "V1":
        return Dense(structured, "cabeceras", "V1", "Chunking por cabeceras y tablas completas + denso BGE-M3", "03")
    if version == "V2":
        return BM25(structured)
    if version == "V3":
        return Hybrid(structured, "cabeceras")
    if version == "V4":
        return Hybrid(structured, "cabeceras", version="V4", technique="V3 + reranker bge-reranker-v2-m3 (top-50)",
                       notebook="05", rerank=True)
    enriched = enrich(structured, document_title(pdf.clean_pages()))
    if version == "V5":
        return Hybrid(enriched, "enriquecidos", version="V5", notebook="05", rerank=True,
                       technique="V4 + metadatos (ruta de sección, cabecera de contexto, tablas serializadas)")
    if version == "V6":
        return QueryTransform(enriched, "enriquecidos", mode=v6_mode)
    raise ValueError(f"Versión desconocida: {version}")
