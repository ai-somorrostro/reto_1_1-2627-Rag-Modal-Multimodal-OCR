# Servicio opcional de ColQwen / ColPali

Servicio FastAPI que indexa las páginas de un PDF con un modelo de recuperación visual por parches
(interacción tardía tipo ColBERT) y responde consultas. Los notebooks lo usan si `COLPALI_URL` está en `.env`
(notebook 7, apartado c, y `casa/casa_04_documento_propio`). Si no está, lo indican y siguen.

Referencia: ColPali, https://arxiv.org/abs/2407.01449

El modelo se carga con `sentence_transformers.MultiVectorEncoder`. Según la ficha de `vidore/colqwen2.5-v0.2`, así se
reproduce el formato de consulta con el que se entrenó; `colpali-engine` ≥ 0.3.13 ya no añade ese prefijo.

Modelo por defecto: `vidore/colqwen2-v1.0`, el único que he podido probar de principio a fin (servicio, API y
cliente de `rag_lab`). `vidore/colqwen2.5-v0.2` se puede elegir con `COLPALI_MODEL`, pero no está probado aquí.
Al cargar ColQwen2 aparecen avisos `custom_text_proj... UNEXPECTED`; la recuperación es correcta igualmente
(ver la tabla de abajo).

## Arranque (servidor con GPU)

```bash
cd servidor/colpali
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
cp ../../data/documento.pdf .
CUDA_VISIBLE_DEVICES=0 PDF_PATH=documento.pdf COLPALI_MODEL=vidore/colqwen2-v1.0 \
    .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8001
```

Variables: `COLPALI_MODEL` (por defecto `vidore/colqwen2-v1.0`), `PDF_PATH`, `DPI` (110).

En el `.env` de los alumnos: `COLPALI_URL=http://IP_DEL_SERVIDOR:8001`.

API:
- `GET /health` -> `{"status": "ok", "model": ..., "pages": 20, "gpu_gb_used": ..., "gpu_gb_peak": ...}`
- `POST /search` con `{"query": "...", "k": 10}` -> `{"results": [{"page": 13, "score": 14.2}, ...]}`

## Memoria de GPU

```bash
.venv/bin/python medir_memoria.py documento.pdf [modelo] [batch]
```

Medido el 2026-10-14 en una RTX 4050 Laptop (6 GB) con `vidore/colqwen2-v1.0` en bf16, las 20 páginas del PDF de clase
a 110 ppp y batch 4 (`torch.cuda.max_memory_allocated`):

| Fase | Pico de memoria | Tiempo |
|---|---|---|
| Carga del modelo | 4,19 GB | 393 s (incluye la descarga) |
| Indexado de 20 páginas | 4,77 GB | 90 s |
| Una consulta | 4,20 GB | 446 ms |
| Reservado por PyTorch al final | 5,44 GB | |

Cabe en 6 GB, pero sin margen para nada más: en el portátil del alumno no puede convivir con los modelos de los
notebooks, por eso se sirve desde el servidor.

`vidore/colqwen2.5-v0.2`: no lo he podido medir. Su modelo base (Qwen2.5-VL-3B) tiene 3.754.885.248 parámetros en bf16
según su ficha en Hugging Face, unos 7,0 GiB solo de pesos, así que no cabe en 6 GB. Hay que medirlo en el A800 con
`medir_memoria.py documento.pdf vidore/colqwen2.5-v0.2 4`.

Calidad con ColQwen2 sobre el PDF de clase: con las 6 preguntas visuales de `data/preguntas.jsonl`, la página correcta
sale en primera posición en las 6.
