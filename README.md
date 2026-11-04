# Semana 2: RAG por versiones

Construimos un RAG en español sobre `data/documento.pdf` y lo mejoramos por versiones (V0 a V7), midiendo cada
cambio. El código común está en `rag_lab/`.

## Puesta en marcha

Requisitos: GPU NVIDIA de 6 GB (o `FORCE_CPU=1` en `.env`), Python 3.11 o 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # rellena OPENROUTER_API_KEY y los modelos
.venv/bin/python preparar_cache.py --casa --anexos    # el profesor, antes de la clase
```

Langfuse es opcional: sin él, las trazas van a `trazas/trazas.jsonl`.

## Clase

| Notebook | Duración | Dificultad | Qué se ve |
|---|---|---|---|
| `01_embeddings` | 1 h | Básica | Embeddings, similitud y búsqueda semántica |
| `02_preparacion_datos` | 1 h | Básica | Docling, chunking, V0 y V1 |
| `03_evaluacion` | 1 h 15 min | Media | recall@k, MRR y LLM-as-a-judge |
| `04_sparse_dense_hibrido` | 1 h 30 min | Media | BM25, denso y RRF: V2 y V3 |
| `05_mejoras_encadenadas` | 1 h 30 min | Media | Reranker, metadatos y multi-query: V4 a V6 |
| `06_ocr_pdf_escaneado` | 1 h | Media | OCR de un PDF escaneado, de la tabla a JSON |
| `07_rag_multimodal` | 1 h 30 min | Alta | VLM, descripciones de página y SigLIP 2: V7 |
| `08_cierre` | 30 min | Básica | Comparar versiones y elegir una |

Las soluciones de los `TODO` están en `soluciones/`.

## Ampliación

- **`casa/`**: observabilidad, análisis de fallos, tu propio PDF y evaluación con Ragas (la más difícil).
- **`anexos/`** (dificultad alta): A1 logo repetido · A2 TF-IDF, convexa y Qdrant · A3 nDCG y calibración del
  judge · A4 CLIP, ColPali y el codificador de imagen.
- **`../semana2-alternativa/`**: `rag-hf-multimodal` (agente con LangChain), `ocr` (más motores de OCR) y
  `rag-advanced` (lo mismo que el 04 y el 05 con LangChain, más Parent Document y compresión).
