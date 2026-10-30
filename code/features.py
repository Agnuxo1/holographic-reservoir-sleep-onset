"""Per-window EEG features for the sleep-onset track.

One 5 s Muse window is (4, 640) at 128 Hz: channels TP9, AF7, AF8, TP10,
unfiltered, in microvolts. The feature set is the classical sleep-staging
one (log band powers plus ratios and Hjorth descriptors), computed with a
Welch periodogram so it stays cheap enough to run on CPU for every window
of 540 recordings.
"""

import numpy as np

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
