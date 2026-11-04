# Qdrant en local (base de datos vectorial)

Base de datos vectorial como servidor para la sección 9 del notebook 3: el mismo híbrido de V3 (BGE-M3 + BM25 con
RRF) hecho dentro de Qdrant, con filtros por metadatos. Sirve para comparar con Chroma (embebida, notebook 2) y con
Elasticsearch (otra asignatura).

Imagen oficial `qdrant/qdrant:v1.19.2` (publicada el 2026-10-05) y cliente de Python `qdrant-client` 1.19.1.
Documentación de consultas híbridas: https://qdrant.tech/documentation/concepts/hybrid-queries/

## Levantarlo

```bash
cd servidor/qdrant
docker compose up -d
curl http://localhost:6333/healthz        # healthz check passed
```

- Panel web: http://localhost:6333/dashboard (colecciones, puntos y payload).
- En `.env`: `QDRANT_URL=http://localhost:6333` (es el valor por defecto).
- Para pararlo: `docker compose down`. Los datos quedan en el volumen `qdrant_storage`; `docker compose down -v` lo borra.

Medido el 2026-10-15: unos 126 MB de RAM con la colección del notebook (63 puntos) y 286 MB de imagen. No usa la GPU.
Si no está en marcha, el notebook 3 lo indica y se salta esa sección.

## Qué hace el notebook

- Cada punto lleva un vector denso (BGE-M3, 1024 dimensiones, coseno), un vector disperso con los pesos BM25 que ya
  calcula `bm25s` y un payload con `paginas`, `seccion` y `tipo`. Ver `rag_lab/vector_db.py`.
- La consulta híbrida usa dos `Prefetch` (denso y disperso) y `RrfQuery`. Qdrant cuenta las posiciones desde 0, así
  que su `k = 61` equivale al `k = 60` de nuestro RRF; su valor por defecto es `k = 2`.
- Con 10 candidatos por lista, las puntuaciones RRF coinciden con las de V3 hecho a mano en las 20 preguntas del PDF;
  el orden solo cambia entre fragmentos empatados.
