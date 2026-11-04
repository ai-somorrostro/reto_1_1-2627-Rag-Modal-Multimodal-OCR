"""Trazas de cada paso del RAG: Langfuse si está configurado; si no, un fichero JSONL local.

Uso:
    with trace("V3", question):        # una traza por pregunta y versión
        frags = retrieve(question)    # funciones decoradas con @traced("retrieve"), etc.

Se registra latencia, tokens, coste e identificadores de los fragmentos de cada paso.
Cada traza lleva el usuario y la etiqueta del alumno (CFG.STUDENT), útiles si varios alumnos comparten instancia.
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import CFG

OBSERVATION_TYPES = {"retrieve": "retriever", "rerank": "retriever", "generate": "generation", "judge": "evaluator",
         "query_transform": "generation"}
EVENTS: list[dict] = []  # todos los pasos de esta sesión, también con Langfuse
_TRACE = contextvars.ContextVar("traza", default=None)
_STATE = {"backend": None, "client": None}


def _host() -> str:
    return os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or ""


def _init() -> None:
    if _STATE["backend"]:
        return
    pk, sk, host = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"), _host()
    if pk and sk and host:
        try:
            from langfuse import Langfuse

            client = Langfuse(public_key=pk, secret_key=sk, base_url=host)
            if client.auth_check():
                _STATE.update(backend="langfuse", client=client)
                return
            warnings.warn("Langfuse rechazó las claves: las trazas se guardan en JSONL.")
        except Exception as e:  # noqa: BLE001 - el tracing nunca debe romper el notebook
            warnings.warn(f"No se pudo conectar con Langfuse ({e}): las trazas se guardan en JSONL.")
    _STATE["backend"] = "jsonl"


def backend() -> str:
    _init()
    return "langfuse " + _host() if _STATE["backend"] == "langfuse" else f"jsonl {jsonl_path()}"


def jsonl_path():
    CFG.TRACES_DIR.mkdir(parents=True, exist_ok=True)
    return CFG.TRACES_DIR / "trazas.jsonl"


def _write_jsonl(event: dict) -> None:
    with open(jsonl_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


@contextmanager
def trace(version: str, question: str, **extra_metadata):
    """Abre una traza para responder a `question` con la versión `version`."""
    _init()
    info = {"trace_id": uuid.uuid4().hex, "version": version, "pregunta": question, "alumno": CFG.STUDENT}
    token = _TRACE.set(info)
    try:
        if _STATE["backend"] == "langfuse":
            from langfuse import propagate_attributes

            lf = _STATE["client"]
            meta = {"version": version, "documento": CFG.DOC_ID, **{k: str(v) for k, v in extra_metadata.items()}}
            with lf.start_as_current_observation(as_type="span", name="responder", input={"pregunta": question}) as root_span:
                with propagate_attributes(user_id=CFG.STUDENT, tags=[f"alumno:{CFG.STUDENT}", version],
                                          trace_name=f"rag-{version}", version=version, metadata=meta):
                    info["trace_id"] = lf.get_current_trace_id()
                    yield info
                root_span.update(output=info.get("salida"))
        else:
            yield info
    finally:
        _TRACE.reset(token)


def _extract(result) -> dict:
    """Saca tokens, coste e ids de fragmentos del valor devuelto por un paso."""
    from .llm import LLMResponse

    payload = {"tokens": 0, "tokens_entrada": 0, "tokens_salida": 0, "coste_usd": 0.0, "ids": [],
             "modelo": None, "latencia_llm_ms": None, "cacheada": False}
    responses = []
    if isinstance(result, LLMResponse):
        responses = [result]
    elif isinstance(result, dict):
        responses = [r for r in result.get("llm", []) if isinstance(r, LLMResponse)]
    elif isinstance(result, list) and result and isinstance(result[0], dict) and "id" in result[0]:
        payload["ids"] = [str(x["id"]) for x in result]
    for r in responses:
        payload["tokens"] += r.tokens
        payload["coste_usd"] += r.cost_usd
        payload["modelo"] = r.model
        payload["latencia_llm_ms"] = (payload["latencia_llm_ms"] or 0) + r.latency_ms
        payload["tokens_entrada"] += r.input_tokens
        payload["tokens_salida"] += r.output_tokens
        payload["cacheada"] = payload["cacheada"] or r.cached
    return payload


def traced(name: str):
    """Decorador para los pasos retrieve, rerank, generate y judge."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _init()
            t0 = time.perf_counter()
            obs = None
            if _STATE["backend"] == "langfuse":
                ctx = _STATE["client"].start_as_current_observation(as_type=OBSERVATION_TYPES.get(name, "span"), name=name)
                obs = ctx.__enter__()
            try:
                result = func(*args, **kwargs)
            except BaseException:
                if obs is not None:
                    ctx.__exit__(*__import__("sys").exc_info())
                raise
            latency = (time.perf_counter() - t0) * 1000
            payload = _extract(result)
            # Si la llamada al LLM vino de la caché, se usa la latencia de la llamada original
            if payload.get("cacheada") and payload["latencia_llm_ms"]:
                latency = latency + payload["latencia_llm_ms"]
            info = _TRACE.get() or {}
            event = {
                "fecha": datetime.now(timezone.utc).isoformat(), "trace_id": info.get("trace_id"),
                "version": info.get("version"), "alumno": CFG.STUDENT, "paso": name,
                "latencia_ms": round(latency, 2), "tokens": payload["tokens"], "coste_usd": payload["coste_usd"],
                "ids_fragmentos": payload["ids"], "modelo": payload["modelo"], "pregunta": info.get("pregunta"),
            }
            EVENTS.append(event)
            if obs is not None:
                try:
                    update_fields = {"metadata": {"latencia_ms": event["latencia_ms"], "ids_fragmentos": payload["ids"]}}
                    if payload["ids"]:
                        update_fields["output"] = {"ids_fragmentos": payload["ids"]}
                    if payload["modelo"]:
                        update_fields.update(
                            model=payload["modelo"],
                            usage_details={"input": payload["tokens_entrada"], "output": payload["tokens_salida"]},
                            cost_details={"total": payload["coste_usd"]},
                        )
                    obs.update(**update_fields)
                finally:
                    ctx.__exit__(None, None, None)
            else:
                _write_jsonl(event)
            return result

        return wrapper

    return decorator


def events_df():
    """Pasos registrados en esta sesión como DataFrame."""
    import pandas as pd

    return pd.DataFrame(EVENTS)


def flush() -> None:
    if _STATE["backend"] == "langfuse":
        _STATE["client"].flush()
