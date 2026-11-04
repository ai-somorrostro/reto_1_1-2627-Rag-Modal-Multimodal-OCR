"""Tabla acumulativa de resultados (results.csv) y gráfico comparativo de versiones."""
from __future__ import annotations

import pandas as pd

from .config import CFG

COLUMNS = [
    "version", "notebook", "tecnica", "corpus",
    "recall@5", "recall@10", "MRR@10",
    "fidelidad", "relevancia_respuesta", "relevancia_contexto",
    "latencia_p50_ms", "latencia_p95_ms", "tokens", "coste_usd",
    "recall@5_texto", "recall@5_tabla", "recall@5_visual",
]

BLUE = "#2a78d6"
TEXT_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def load() -> pd.DataFrame:
    if CFG.RESULTS_CSV.exists():
        return pd.read_csv(CFG.RESULTS_CSV)
    return pd.DataFrame(columns=COLUMNS)


def register(row: dict) -> pd.DataFrame:
    """Añade o reemplaza la fila de una versión (por `version` y `corpus`) y guarda el CSV."""
    df = load()
    row = {c: row.get(c) for c in COLUMNS}
    mask = (df["version"] == row["version"]) & (df["corpus"] == row["corpus"])
    df = pd.concat([df[~mask], pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["corpus", "version"], key=lambda s: s.astype(str)).reset_index(drop=True)
    df.to_csv(CFG.RESULTS_CSV, index=False)
    return df


def table(corpus: str | None = None) -> pd.DataFrame:
    df = load()
    if corpus:
        df = df[df["corpus"] == corpus]
    return df.set_index("version")


def plot_comparison(corpus: str = "pdf", path=None, metrics=None):
    """Paneles pequeños: una métrica por panel, una barra por versión."""
    import matplotlib.pyplot as plt

    metrics = metrics or [
        ("recall@5", "Recall@5"), ("MRR@10", "MRR@10"), ("fidelidad", "Fidelidad (judge)"),
        ("latencia_p50_ms", "Latencia p50 (ms)"), ("coste_usd", "Coste total (USD)"),
    ]
    df = table(corpus)
    metrics = [(c, t) for c, t in metrics if c in df and df[c].notna().any()]
    fig, panels = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.45 * len(df) + 1.6), sharey=True)
    panels = [panels] if len(metrics) == 1 else panels
    for panel, (col, title) in zip(panels, metrics):
        vals = df[col].astype(float)
        panel.barh(df.index, vals, color=BLUE, height=0.6)
        for y, v in enumerate(vals):
            if pd.notna(v):
                label = f" {v:,.0f}".replace(",", ".") if v >= 100 else f" {v:.3g}"
                panel.text(v, y, label, va="center", fontsize=8, color=TEXT_SECONDARY)
        panel.set_title(title, fontsize=10, color=TEXT_COLOR, loc="left")
        panel.grid(axis="x", color=GRID, linewidth=0.8)
        panel.set_axisbelow(True)
        panel.margins(x=0.25)
        for side in ("top", "right", "left"):
            panel.spines[side].set_visible(False)
        panel.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    panels[0].invert_yaxis()
    fig.suptitle(f"Comparación de versiones del RAG (corpus: {corpus})", x=0.01, ha="left", fontsize=11, color=TEXT_COLOR)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
