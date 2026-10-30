"""Submission for Track 03 (Sleep onset) - EEG/EMG Foundation Challenge 2026.

Window features plus the state of a Holographic-Reservoir
(sparse expander graph with SHA-256 derived entropy on its
diagonal) and the time elapsed since the recording started.
Measured as a tie with the explicit-memory model; shipped for the
record, not as the better system.

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
SFREQ = 128.0
BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)
CLIP_UV = 500.0  # Muse windows carry eye-blink and movement spikes.


def _welch(x, sfreq=SFREQ, nperseg=256):
    """Periodogram averaged over Hann segments: x (..., T) -> (freqs, psd)."""
    n_times = x.shape[-1]
    nperseg = min(nperseg, n_times)
    step = nperseg // 2
    starts = range(0, n_times - nperseg + 1, step) if n_times >= nperseg else [0]
    window = np.hanning(nperseg)
    scale = 1.0 / (sfreq * (window ** 2).sum())
    segments = []
    for start in starts:
        seg = x[..., start:start + nperseg]
        seg = (seg - seg.mean(axis=-1, keepdims=True)) * window
        spec = np.fft.rfft(seg, axis=-1)
        psd = (np.abs(spec) ** 2) * scale
        psd[..., 1:-1] *= 2.0
        segments.append(psd)
    psd = np.mean(segments, axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    return freqs, psd


def window_features(x):
    """Features of one batch of windows: (B, C, T) -> (B, F) float64.

    Amplitudes are clipped before anything else: the recordings are raw, and
    a single blink otherwise dominates every power in the window.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    x = np.clip(x, -CLIP_UV, CLIP_UV)
    x = x - x.mean(axis=-1, keepdims=True)

    freqs, psd = _welch(x)                      # (B, C, F)
    powers = []
    for _name, low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        powers.append(psd[..., mask].sum(axis=-1))
    powers = np.stack(powers, axis=-1)          # (B, C, n_bands)
    total = powers.sum(axis=-1, keepdims=True) + 1e-12

    log_powers = np.log10(powers + 1e-12)       # absolute level
    rel_powers = powers / total                 # spectral shape

    # Ratios that track drowsiness: alpha drops and theta rises into N1/N2.
    alpha = powers[..., 2]
    theta = powers[..., 1]
    delta = powers[..., 0]
    sigma = powers[..., 3]
    ratios = np.stack([
        np.log10((theta + 1e-12) / (alpha + 1e-12)),
        np.log10((delta + 1e-12) / (alpha + 1e-12)),
        np.log10((sigma + 1e-12) / (alpha + 1e-12)),
        np.log10((delta + theta + 1e-12) / (alpha + sigma + 1e-12)),
    ], axis=-1)

    # Hjorth activity / mobility / complexity, and a saturation flag.
    d1 = np.diff(x, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0 = x.var(axis=-1) + 1e-12
    var1 = d1.var(axis=-1) + 1e-12
    var2 = d2.var(axis=-1) + 1e-12
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    hjorth = np.stack([
        np.log10(var0),
        mobility,
        complexity,
        (np.abs(x) >= CLIP_UV - 1e-9).mean(axis=-1),
    ], axis=-1)

    feats = np.concatenate([
        log_powers.reshape(len(x), -1),
        rel_powers.reshape(len(x), -1),
        ratios.reshape(len(x), -1),
        hjorth.reshape(len(x), -1),
    ], axis=-1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


N_FEATURES = len(BANDS) * 4 * 2 + 4 * 4 + 4 * 4
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

    name = "HolographicReservoir-Sleep"

    def load_model(self, meta):
        weights = np.load(meta["submission_dir"] / "weights.npz")
        return SleepOnsetModel(weights)
