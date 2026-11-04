"""Mide la memoria de GPU que necesita el servicio: carga del modelo, indexado del PDF y una consulta.

Uso: python medir_memoria.py documento.pdf [modelo] [batch_size]
"""
import sys
import time

import torch
from sentence_transformers import MultiVectorEncoder

from app import pages_as_images

pdf = sys.argv[1]
model_id = sys.argv[2] if len(sys.argv) > 2 else "vidore/colqwen2-v1.0"
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 4
gb = lambda: torch.cuda.max_memory_allocated() / 2**30

print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
t = time.time()
model = MultiVectorEncoder(model_id, device="cuda", model_kwargs={"dtype": torch.bfloat16})
print(f"Modelo {model_id} cargado: pico {gb():.2f} GB ({time.time() - t:.0f} s)")
images = pages_as_images(pdf, 110)
torch.cuda.reset_peak_memory_stats()
t = time.time()
with torch.no_grad():
    docs = model.encode_document(images, batch_size=batch)
print(f"Indexadas {len(images)} páginas (batch {batch}): pico {gb():.2f} GB ({time.time() - t:.0f} s)")
torch.cuda.reset_peak_memory_stats()
t = time.time()
with torch.no_grad():
    q = model.encode_query(["¿En qué meses se ejecuta la fase 3 del cronograma?"])
    scores = model.similarity(q, docs)[0]
print(f"Consulta: pico {gb():.2f} GB ({(time.time() - t) * 1000:.0f} ms); mejor página {int(scores.argmax()) + 1}")
print(f"Memoria reservada por PyTorch al final: {torch.cuda.memory_reserved() / 2**30:.2f} GB")
