"""LLM-as-a-judge con la tríada del RAG y su calibración frente a etiquetas humanas.

Cada criterio se juzga por separado con un veredicto binario (1 = sí, 0 = no) y un razonamiento previo.
El veredicto binario facilita que el profesor etiquete y que el acuerdo se mida con kappa de Cohen.
"""
from __future__ import annotations

import pandas as pd

from . import llm
from .config import CFG
from .tracing import traced

CRITERIA = ["fidelidad", "relevancia_respuesta", "relevancia_contexto"]

SYSTEM = "Eres un evaluador estricto de sistemas RAG. Respondes solo con el JSON pedido, en español."

PROMPTS = {
    "fidelidad": (
        "Evalúa la FIDELIDAD de la respuesta al contexto.\n"
        "Veredicto 'si' cuando todas las afirmaciones de la respuesta (cifras, fechas, nombres) aparecen en el contexto "
        "o se deducen de él directamente. También 'si' cuando la respuesta dice que el contexto no contiene la información. "
        "Veredicto 'no' si alguna afirmación no está respaldada por el contexto o lo contradice.\n\n"
        "CONTEXTO:\n{context}\n\nRESPUESTA:\n{answer}"
    ),
    "relevancia_respuesta": (
        "Evalúa la RELEVANCIA DE LA RESPUESTA respecto a la pregunta.\n"
        "Veredicto 'si' cuando la respuesta contesta directamente a lo que se pregunta. "
        "Veredicto 'no' si es evasiva, responde a otra cosa o dice que no puede responder.\n"
        "No juzgues si es correcta, solo si responde a la pregunta.\n\n"
        "PREGUNTA:\n{question}\n\nRESPUESTA:\n{answer}"
    ),
    "relevancia_contexto": (
        "Evalúa la RELEVANCIA DEL CONTEXTO recuperado respecto a la pregunta.\n"
        "Veredicto 'si' cuando el contexto contiene la información necesaria para responder a la pregunta. "
        "Veredicto 'no' si falta esa información, aunque el contexto trate un tema parecido.\n\n"
        "PREGUNTA:\n{question}\n\nCONTEXTO:\n{context}"
    ),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "razonamiento": {"type": "string", "description": "Justificación breve, antes del veredicto"},
        "veredicto": {"type": "string", "enum": ["si", "no"]},
    },
    "required": ["razonamiento", "veredicto"],
    "additionalProperties": False,
}


def judge_criterion(criterion: str, question: str, answer_text: str, context: str) -> tuple[int, str, llm.LLMResponse]:
    prompt = PROMPTS[criterion].format(question=question, answer=answer_text, context=context)
    r = llm.judge(prompt, SCHEMA, system=SYSTEM)
    payload = r.json()
    return int(payload["veredicto"] == "si"), payload["razonamiento"], r


@traced("judge")
def judge_triad(question: str, answer_text: str, context: str) -> dict:
    """Devuelve {criterio: 0/1, "razones": {...}, "llm": [Respuesta, ...]}."""
    output = {"razones": {}, "llm": []}
    for criterion in CRITERIA:
        verdict, reason, r = judge_criterion(criterion, question, answer_text, context)
        output[criterion] = verdict
        output["razones"][criterion] = reason
        output["llm"].append(r)
    return output


# --- Calibración -------------------------------------------------------------------------------

HUMAN_COLUMNS = [f"{c}_humano" for c in CRITERIA]


def create_label_template(items: list[dict], path=None) -> bool:
    """Crea el CSV de etiquetas humanas con las columnas de etiqueta vacías. Nunca sobrescribe."""
    path = path or CFG.LABELS_PATH
    if path.exists():
        return False
    df = pd.DataFrame(items)[["id", "version", "pregunta", "respuesta", "contexto"]]
    for col in HUMAN_COLUMNS:
        df[col] = ""
    df.to_csv(path, index=False)
    return True


def load_labels(path=None) -> pd.DataFrame | None:
    """Devuelve las filas etiquetadas por completo, o None si el fichero aún está vacío."""
    path = path or CFG.LABELS_PATH
    if not path.exists():
        print(f"No existe {path}: se continúa sin calibración.")
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    valid = pd.Series(True, index=df.index)
    for col in HUMAN_COLUMNS:
        df[col] = df[col].str.strip()
        valid &= df[col].isin(["0", "1"])
    complete = df[valid].copy()
    if complete.empty:
        print(f"{path.name} todavía no tiene etiquetas humanas (columnas *_humano vacías). "
              "Se continúa sin calibración: rellénalas con 1 (sí) o 0 (no) y vuelve a ejecutar esta celda.")
        return None
    print(f"Etiquetas humanas: {len(complete)} de {len(df)} filas completas.")
    return complete.astype({c: int for c in HUMAN_COLUMNS})


def agreement(judge: list[int], human: list[int]) -> dict:
    """Porcentaje de acuerdo y kappa de Cohen entre judge y etiquetas humanas."""
    from sklearn.metrics import cohen_kappa_score

    matches = sum(int(a == b) for a, b in zip(judge, human))
    kappa = cohen_kappa_score(human, judge) if len(set(human) | set(judge)) > 1 else float("nan")
    return {"n": len(human), "acuerdo_%": 100 * matches / len(human), "kappa": kappa}
