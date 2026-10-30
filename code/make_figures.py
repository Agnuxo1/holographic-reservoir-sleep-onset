"""Figures for the report and the public README, from measured results only.

Two panels, both from real runs:

1. The ladder: W-bMAE per model, mean over folds and seeds, with the spread
   across folds, against the organisers' published reference scores.
2. Why it works: median true latency and each model's median prediction as a
   function of elapsed time inside the recording, on held-out recordings.
   A stateless model is a flat line by construction; the truth is a ramp.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

PUBLISHED = {"Constante 35 s": 60.7, "EEGNet": 60.6, "CTNet": 60.1,
             "REVE afinado": 72.9}
LABELS = {
    "A_constante": "A  constante",
    "B_ridge_sin_estado": "B  ridge sin estado",
    "C_solo_reloj": "C  solo reloj",
    "D_reloj_mas_ventana": "D  reloj + ventana",
    "E_reservorio_holografico": "E  reservorio holografico",
    "F_reservorio_esn": "F  reservorio ESN",
    "G_reservorio_sin_reloj": "G  reservorio sin reloj",
    "H_reservorio_solo_estado": "H  reservorio solo estado",
    "I_memoria_explicita_10s": "I  memoria explicita 10 s",
    "J_reservorio_residual": "J  reservorio residual",
}
# The submitted system, drawn apart from the rest.
CHOSEN = "I_memoria_explicita_10s"
INK = "#1b1b1f"
MUTED = "#8a8a94"
ACCENT = "#1f6feb"
WARM = "#c2410c"


def ladder(results, regime, ax):
    import matplotlib.pyplot as plt  # noqa: F401
    names = [n for n in LABELS if any(
        r["model"] == n and r["regime"] == regime for r in results)]
    means, spreads = [], []
    for name in names:
        vals = [r["wbmae"] for r in results
                if r["model"] == name and r["regime"] == regime]
        means.append(np.mean(vals))
        spreads.append(np.std(vals))
    y = np.arange(len(names))
    colors = []
    for n in names:
        if n == CHOSEN:
            colors.append(INK)
        elif n.startswith(("E_", "F_", "G_", "H_", "J_")):
            colors.append(ACCENT)
        else:
            colors.append(MUTED)
    ax.barh(y, means, xerr=spreads, color=colors, height=0.62,
            error_kw=dict(ecolor=INK, lw=1.0, capsize=3, alpha=0.65))
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[n] for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(PUBLISHED["CTNet"], color=WARM, lw=1.1, ls="--", alpha=0.85)
    ax.text(PUBLISHED["CTNet"] - 0.6, len(names) - 1.6,
            "mejor control publicado\n(CTNet, 60,1)", color=WARM,
            fontsize=7.5, ha="right", va="center")
    for yi, (mean, spread) in enumerate(zip(means, spreads)):
        ax.text(mean + spread + 0.8, yi, f"{mean:.1f}", va="center",
                fontsize=8.5, color=INK)
    ax.set_xlabel("W-bMAE (s)  -  menor es mejor", fontsize=9)
    ax.set_title(f"Escalera de modelos, regimen {regime}", fontsize=10.5,
                 loc="left", color=INK, pad=10)
    ax.set_xlim(0, max(68.0, max(means) + 8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)


def profile(ax):
    """Median truth and median prediction against elapsed time."""
    from models import (ElapsedQuantile, EmaRidge,  # noqa: E402
                        ReservoirStream, StatelessRidge)
    from run_cv import folds_unseen, load_cache, subset

    data = load_cache("train")
    mask = next(folds_unseen(data, 5, seed=0))
    train, test = subset(data, ~mask), subset(data, mask)
    order = np.lexsort((test["elapsed"], test["record_id"]))
    test = {k: v[order] for k, v in test.items()}

    bins = np.arange(0, 1205, 60.0)
    centres = bins[:-1] + 30.0
    idx = np.digitize(test["elapsed"], bins) - 1
    keep = (idx >= 0) & (idx < len(centres))

    def median_by_bin(values):
        out = np.full(len(centres), np.nan)
        for b in range(len(centres)):
            sel = keep & (idx == b)
            if sel.sum() >= 20:
                out[b] = np.median(values[sel])
        return out

    ax.plot(centres, median_by_bin(test["y"]), color=INK, lw=2.2,
            label="verdad (mediana)", zorder=5)
    for factory, label, color, style in (
            (lambda: StatelessRidge(alpha=30.0), "B  ridge sin estado",
             MUTED, "--"),
            (lambda: ElapsedQuantile(), "C  solo reloj", WARM, "-."),
            (lambda: ReservoirStream(size=100, leak=0.7, alpha=30.0, seed=0),
             "E  reservorio", ACCENT, "-"),
            (lambda: EmaRidge(alpha=10.0, halflives=(2,)),
             "I  memoria explicita", "#15803d", "-")):
        model = factory().fit(train)
        pred = model.predict_feats(test["feats"], test["elapsed"],
                                    test["record_id"])
        ax.plot(centres, median_by_bin(np.asarray(pred)), color=color,
                lw=1.7, ls=style, label=label)

    ax.set_xlabel("tiempo transcurrido en la grabacion (s)", fontsize=9)
    ax.set_ylabel("latencia predicha (s)", fontsize=9)
    ax.set_title("Por que un modelo sin estado no puede acertar", fontsize=10.5,
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/cv_results.json")
    parser.add_argument("--out", default="repo/figures")
    args = parser.parse_args()

    results = json.loads((ROOT / args.results).read_text(encoding="utf-8"))
    regimes = sorted({r["regime"] for r in results})

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#c9c9d1"})
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.6 * len(regimes), 4.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, regime in zip(axes, regimes):
        ladder(results, regime, ax)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "escalera.png", bbox_inches="tight")
    print("escrito", out_dir / "escalera.png")

    fig2, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    profile(ax)
    fig2.savefig(out_dir / "perfil_temporal.png", bbox_inches="tight")
    print("escrito", out_dir / "perfil_temporal.png")


if __name__ == "__main__":
    main()
