"""Genera antes de la clase todas las cachés que usan los notebooks (Docling, páginas, embeddings, LLM, judge, VLM).

Ejecuta cada notebook en una copia dentro de cache/ejecuciones/, así las salidas entregadas no se tocan y las
cachés quedan exactamente con las llamadas que harán los alumnos.

Uso:
    python preparar_cache.py                 # notebooks de clase (01 a 08), FAST_MODE según .env
    python preparar_cache.py --casa          # además, los notebooks de casa
    python preparar_cache.py --anexos        # además, los anexos opcionales
    python preparar_cache.py --completo      # FAST_MODE=0 (corpus de control completo)
    python preparar_cache.py --solo 03 04    # solo los notebooks indicados (prefijo del nombre)
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASS_NOTEBOOKS = sorted(p for p in ROOT.glob("0*.ipynb"))
HOME_NOTEBOOKS = sorted((ROOT / "casa").glob("casa_*.ipynb"))
ANNEX_NOTEBOOKS = sorted((ROOT / "anexos").glob("A*.ipynb"))


def cached_calls() -> tuple[int, float]:
    path = ROOT / "cache" / "llm.sqlite"
    if not path.exists():
        return 0, 0.0
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT json_extract(respuesta, '$.cost_usd') FROM llm").fetchall()
    return len(rows), sum(f[0] or 0 for f in rows)


def run_notebook(notebook: Path, target: Path, env_vars: dict) -> bool:
    target.mkdir(parents=True, exist_ok=True)
    nb_copy = target / notebook.name
    shutil.copy(notebook, nb_copy)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=3600", str(nb_copy)],
                       cwd=notebook.parent, env=env_vars, capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "ERROR"
    print(f"  {notebook.relative_to(ROOT)}: {status} ({time.time() - t0:.0f} s)", flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
    return r.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--casa", dest="home", action="store_true", help="incluir los notebooks de casa")
    parser.add_argument("--anexos", dest="annexes", action="store_true", help="incluir los anexos")
    parser.add_argument("--completo", dest="full", action="store_true", help="FAST_MODE=0")
    parser.add_argument("--solo", dest="only", nargs="*", default=None, help="prefijos de notebooks a ejecutar")
    args = parser.parse_args()

    env_vars = {**os.environ}
    if args.full:
        env_vars["FAST_MODE"] = "0"
    notebooks = CLASS_NOTEBOOKS + (HOME_NOTEBOOKS if args.home else []) + (ANNEX_NOTEBOOKS if args.annexes else [])
    if args.only:
        notebooks = [n for n in notebooks if any(n.name.startswith(p) for p in args.only)]
    if not (ROOT / "data" / "documento.pdf").exists():
        print("Falta data/documento.pdf (o define DOC_PATH en .env).")
        return 1

    if not (ROOT / "data" / "preguntas.jsonl").exists():
        print("No existe data/preguntas.jsonl: se genera el borrador con rag_lab.preguntas (revísalo a mano).")
        subprocess.run([sys.executable, "-c", "from rag_lab import questions; questions.generate_questions()"],
                       cwd=ROOT, env=env_vars, check=True)

    n0, cost0 = cached_calls()
    print(f"Caché del LLM antes: {n0} llamadas. Ejecutando {len(notebooks)} notebooks...")
    t0 = time.time()
    ok = all([run_notebook(nb, ROOT / "cache" / "ejecuciones" / nb.parent.relative_to(ROOT), env_vars) for nb in notebooks])
    n1, cost1 = cached_calls()
    print(f"Hecho en {(time.time() - t0) / 60:.1f} min. Llamadas nuevas al LLM: {n1 - n0} "
          f"(coste {cost1 - cost0:.4f} USD). Total en caché: {n1} llamadas.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
