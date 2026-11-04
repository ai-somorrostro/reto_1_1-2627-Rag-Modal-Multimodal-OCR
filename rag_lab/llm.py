"""Cliente de OpenRouter (API compatible con OpenAI) con caché en disco, coste y reintentos.

OpenRouter devuelve siempre `usage` en las respuestas sin streaming, con `cost` en créditos (USD).
La caché guarda la respuesta junto con los tokens, el coste y la latencia de la llamada original,
de modo que las métricas de coste y latencia no cambian al repetir la ejecución.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sqlite3
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CFG


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def json(self) -> dict:
        """Interpreta el texto como JSON (para salidas estructuradas)."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)


CALL_LOG: list[dict] = []  # una fila por llamada en esta sesión


class MissingKeyError(RuntimeError):
    pass


def family(model: str) -> str:
    """Proveedor o familia del modelo: el prefijo antes de '/' (openai, google, anthropic...)."""
    return model.split("/")[0].lower()


def check_families() -> bool:
    """Avisa si el generador y el judge son de la misma familia (sesgo de autopreferencia)."""
    if family(CFG.GEN_MODEL) == family(CFG.JUDGE_MODEL):
        warnings.warn(
            f"GEN_MODEL ({CFG.GEN_MODEL}) y JUDGE_MODEL ({CFG.JUDGE_MODEL}) son de la misma familia: "
            "el judge puede preferir las respuestas de su propio modelo. Usa otra familia para el judge."
        )
        return False
    return True


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS llm (clave TEXT PRIMARY KEY, respuesta TEXT)")

    def read(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT respuesta FROM llm WHERE clave=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, key: str, payload: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, json.dumps(payload, ensure_ascii=False)))
        self.conn.commit()


_CACHE: _Cache | None = None
_CLIENT = None


def _cache() -> _Cache:
    global _CACHE
    if _CACHE is None:
        _CACHE = _Cache(CFG.CACHE_DIR / "llm.sqlite")
    return _CACHE


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise MissingKeyError("Falta OPENROUTER_API_KEY y la llamada no está en la caché. Configura .env (ver README).")
        _CLIENT = OpenAI(base_url=CFG.OPENROUTER_BASE_URL, api_key=key, max_retries=0, timeout=120)
    return _CLIENT


def _key(model: str, messages: list, params: dict) -> str:
    raw = json.dumps({"modelo": model, "mensajes": messages, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chat(messages: list[dict], model: str, *, temperature: float = 0.0, max_tokens: int = 1024,
         response_format: dict | None = None, reasoning: dict | None = None, retries: int = 5,
         step: str = "") -> LLMResponse:
    """Llamada a chat completions con caché. La clave incluye modelo, mensajes y parámetros."""
    params = {"temperature": temperature, "max_tokens": max_tokens, "seed": CFG.SEED}
    extra_body = {}
    if response_format:
        params["response_format"] = response_format
    if reasoning:
        extra_body["reasoning"] = reasoning
    key = _key(model, messages, {**params, **extra_body})

    stored_row = _cache().read(key)
    if stored_row is not None:
        r = LLMResponse(**{**stored_row, "cached": True})
        CALL_LOG.append({**asdict(r), "paso": step})
        return r

    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    wait_s = 2.0
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = _client().chat.completions.create(model=model, messages=messages, extra_body=extra_body or None, **params)
            latency = (time.perf_counter() - t0) * 1000
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError(f"Respuesta vacía (finish_reason={resp.choices[0].finish_reason})")
            break
        except (RateLimitError, APIConnectionError, APITimeoutError, ValueError) as e:
            error = e
        except APIStatusError as e:
            if e.status_code < 500 and e.status_code not in (408, 429):
                raise
            error = e
        if attempt == retries - 1:
            raise RuntimeError(f"OpenRouter falló tras {retries} intentos: {error}") from error
        time.sleep(wait_s)
        wait_s *= 2

    usage = resp.usage
    r = LLMResponse(
        text=text,
        model=resp.model or model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=float(getattr(usage, "cost", 0.0) or 0.0),
        latency_ms=latency,
    )
    payload = asdict(r)
    payload.pop("cached")
    _cache().save(key, payload)
    CALL_LOG.append({**asdict(r), "paso": step})
    return r


def generate(prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
    """Llamada al modelo generador (GEN_MODEL)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return chat(messages, CFG.GEN_MODEL, step=kwargs.pop("step", "generate"), **kwargs)


def judge(prompt: str, schema: dict, system: str | None = None, **kwargs) -> LLMResponse:
    """Llamada al judge (JUDGE_MODEL) con salida estructurada según un JSON Schema."""
    check_families()
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    fmt = {"type": "json_schema", "json_schema": {"name": "veredicto", "strict": True, "schema": schema}}
    kwargs.setdefault("max_tokens", 4000)  # el judge razona: el límite cubre razonamiento y salida
    kwargs.setdefault("reasoning", {"effort": "low", "exclude": True})
    return chat(messages, CFG.JUDGE_MODEL, response_format=fmt, step=kwargs.pop("step", "judge"), **kwargs)


def _image_to_url(image) -> str:
    """Convierte una ruta o una imagen PIL en una URL data: base64 (PNG)."""
    if isinstance(image, (str, Path)):
        payload = Path(image).read_bytes()
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def vision(prompt: str, images: list, schema: dict | None = None, **kwargs) -> LLMResponse:
    """Llamada al modelo de visión (VLM_MODEL): primero el texto y después las imágenes."""
    content_parts = [{"type": "text", "text": prompt}]
    content_parts += [{"type": "image_url", "image_url": {"url": _image_to_url(img)}} for img in images]
    fmt = None
    if schema:
        fmt = {"type": "json_schema", "json_schema": {"name": "salida", "strict": True, "schema": schema}}
    kwargs.setdefault("max_tokens", 2048)
    return chat([{"role": "user", "content": content_parts}], CFG.VLM_MODEL, response_format=fmt,
                step=kwargs.pop("step", "vision"), **kwargs)


def calls_df():
    """Tabla con todas las llamadas de esta sesión (tokens, coste, latencia, si venían de caché)."""
    import pandas as pd

    return pd.DataFrame(CALL_LOG)
