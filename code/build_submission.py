"""Train the chosen model on all public training data and write the zip.

The zip holds exactly two files:

* ``submission.py`` - a self-contained benchopt solver. The feature code is
  spliced in from ``code/features.py`` at build time, so the submitted file
  and the file the experiments were run with cannot drift apart. Inference
  imports nothing but numpy.
* ``weights.npz`` - the fitted readout, the moving-average timescales and
  the elapsed-time table (plus the reservoir matrices for ``--model E``).

Two models can be built, because the measurements ended in a tie and the
sealed phase allows five submissions:

* ``--model I`` (default): window features, explicit moving averages and the
  clock. The best measured system.
* ``--model E``: the same with the Holographic-Reservoir state in place of
  the moving averages, for the record.

The build prints the offline W-bMAE of the fitted model on the public test
split. ``code/verify_official.py`` then scores the written zip through the
competition's own harness; the two numbers must agree, or the packaging is
wrong rather than the model.

Usage::

    python code/build_submission.py --model I --out submissions/memoria_10s
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models import (EmaRidge, ReservoirStream,  # noqa: E402
                    wbmae_recording_mean)
from run_cv import load_cache  # noqa: E402

TEMPLATE = '''"""Submission for Track 03 (Sleep onset) - EEG/EMG Foundation Challenge 2026.

{headline}

The model follows the streamed contract: ``reset_state()`` at the start of
each recording, then one ``predict`` call per 5 s window, in time order.
Inference needs only numpy.

Author: Francisco Angulo de Lafuente (Agnuxo). Licence: Apache-2.0.
"""

import numpy as np

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0
WINDOW_S = 5.0

# --- feature code, spliced from code/features.py at build time -----------
{features}
# --- end of spliced feature code -----------------------------------------


def _to_numpy(X):
    return X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)


def _elapsed_features(elapsed):
    e = np.asarray(elapsed, float).reshape(-1, 1) / CAP_S
    return np.concatenate([
        np.ones_like(e),
        np.minimum(e, 4.0),
        np.sqrt(np.minimum(e, 16.0)),
        np.log1p(np.minimum(e, 16.0)),
        np.exp(-np.minimum(e, 16.0)),
    ], axis=1)


class SleepOnsetModel:
    """Linear readout over the window, its recent history and the clock."""

    def __init__(self, weights):
        self.kind = str(weights["kind"])
        self.halflives = [float(h) for h in weights["halflives"]]
        self.ridge_mean = weights["ridge_mean"]
        self.ridge_scale = weights["ridge_scale"]
        self.coef = weights["coef"]
        self.prior_edges = weights["prior_edges"]
        self.prior_table = weights["prior_table"]
        self.use_prior = bool(weights["use_prior"])
        if self.kind == "reservoir":
            self.W_in = weights["W_in"]
            self.W = weights["W"]
            self.leak = float(weights["leak"])
            self.mean_in = weights["mean_in"]
            self.scale_in = weights["scale_in"]
            self.size = self.W.shape[0]
        self.reset_state()

    def reset_state(self):
        self.k = 0
        self.ema = None
        if self.kind == "reservoir":
            self.state = np.zeros(self.size)

    def _prior(self, elapsed):
        idx = np.digitize(np.asarray(elapsed, float), self.prior_edges)
        return np.clip(self.prior_table[idx], 0.0, CAP_S)

    def _memory(self, row):
        if self.kind == "reservoir":
            u = (row - self.mean_in) / self.scale_in
            self.state = ((1.0 - self.leak) * self.state
                          + self.leak * np.tanh(self.W_in @ u
                                                + self.W @ self.state))
            return self.state
        if self.ema is None:
            self.ema = [row.copy() for _ in self.halflives]
        for j, halflife in enumerate(self.halflives):
            decay = 0.5 ** (1.0 / halflife)
            self.ema[j] = decay * self.ema[j] + (1.0 - decay) * row
        return np.concatenate(self.ema)

    def predict(self, X):
        feats = window_features(_to_numpy(X))
        out = np.empty(len(feats))
        for i in range(len(feats)):
            row = feats[i]
            memory = self._memory(row)
            elapsed = [self.k * WINDOW_S]
            if self.kind == "reservoir":
                parts = [memory[None], row[None], _elapsed_features(elapsed)]
            else:
                parts = [row[None], memory[None], _elapsed_features(elapsed)]
            if self.use_prior:
                parts.append(self._prior(elapsed).reshape(-1, 1) / CAP_S)
            Z = np.concatenate(parts, axis=1)
            Zs = np.concatenate([(Z - self.ridge_mean) / self.ridge_scale,
                                 np.ones((1, 1))], axis=1)
            out[i] = float(np.clip(Zs @ self.coef, 0.0, CAP_S)[0])
            self.k += 1
        return out


class Solver(CompetSolver):

    name = "{name}"

    def load_model(self, meta):
        weights = np.load(meta["submission_dir"] / "weights.npz")
        return SleepOnsetModel(weights)
'''

HEADLINE = {
    "I": ("Window features (log band powers, spectral ratios and Hjorth\n"
          "descriptors over the four Muse channels), explicit exponential\n"
          "moving averages of those features, and the time elapsed since the\n"
          "recording started. The readout is a linear model fitted to the\n"
          "competition's own weighted absolute error rather than to a squared\n"
          "error, which on this task is worth more than any of the memory\n"
          "mechanisms that were tried."),
    "E": ("Window features plus the state of a Holographic-Reservoir\n"
          "(sparse expander graph with SHA-256 derived entropy on its\n"
          "diagonal) and the time elapsed since the recording started.\n"
          "Measured as a tie with the explicit-memory model; shipped for the\n"
          "record, not as the better system."),
}


def features_source():
    """The body of features.py, minus its imports and module docstring."""
    source = (ROOT / "code" / "features.py").read_text(encoding="utf-8")
    return source[source.index("SFREQ = 128.0"):].rstrip()


def export_weights(model, kind):
    core = model.core
    prior = model.prior
    out = dict(
        kind=np.str_(kind),
        ridge_mean=core.mean, ridge_scale=core.scale, coef=core.coef,
        prior_edges=(prior.edges if prior is not None
                     else np.zeros(1, dtype=float)),
        prior_table=(prior.table if prior is not None
                     else np.zeros(2, dtype=float)),
        use_prior=np.bool_(prior is not None),
        halflives=np.asarray(getattr(model, "halflives", ()), dtype=float),
    )
    if kind == "reservoir":
        out.update(W_in=model.W_in, W=model.W,
                   leak=np.float64(model.leak),
                   mean_in=model.mean_in, scale_in=model.scale_in)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="I", choices=["I", "E"])
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--halflife", type=float, nargs="+", default=[2.0])
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--leak", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    train = load_cache("train")
    print(f"ajustando con {len(train['y'])} ventanas de "
          f"{len(np.unique(train['record_id']))} grabaciones", flush=True)

    if args.model == "I":
        model = EmaRidge(alpha=args.alpha,
                         halflives=tuple(args.halflife))
        kind = "ema"
        name = args.name or "Explicit-Memory-Sleep"
        out_rel = args.out or "submissions/memoria_explicita"
    else:
        model = ReservoirStream(size=args.size, leak=args.leak,
                                alpha=30.0, seed=args.seed,
                                holographic=True)
        kind = "reservoir"
        name = args.name or "HolographicReservoir-Sleep"
        out_rel = args.out or "submissions/reservorio_holografico"
    model.fit(train)

    # Reference number: the fitted model scored offline on the public test
    # split, with my own code. verify_official.py must reproduce it through
    # the competition harness.
    test = load_cache("test")
    order = np.lexsort((test["elapsed"], test["record_id"]))
    test = {k: v[order] for k, v in test.items()}
    pred = model.predict_feats(test["feats"], test["elapsed"],
                               test["record_id"])
    offline = wbmae_recording_mean(test["y"], pred, test["record_id"])
    print(f"W-bMAE offline en el split publico de test: {offline:.3f}",
          flush=True)

    out_dir = ROOT / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "weights.npz",
                        **export_weights(model, kind))
    submission = TEMPLATE.format(features=features_source(), name=name,
                                 headline=HEADLINE[args.model])
    (out_dir / "submission.py").write_text(submission, encoding="utf-8")

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "submission.py", "submission.py")
        zf.write(out_dir / "weights.npz", "weights.npz")

    meta = dict(model=args.model, kind=kind, name=name, alpha=args.alpha,
                halflives=args.halflife, size=args.size, leak=args.leak,
                seed=args.seed,
                n_train_windows=int(len(train["y"])),
                n_train_records=int(len(np.unique(train["record_id"]))),
                wbmae_offline_public_test=round(offline, 3),
                zip_bytes=zip_path.stat().st_size)
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=1),
                                             encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print("zip:", zip_path)


if __name__ == "__main__":
    main()
