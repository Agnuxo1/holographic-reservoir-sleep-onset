"""Streaming models for the sleep-onset track, from the floor up to the
holographic reservoir.

Every model follows the track's contract: ``reset_state()`` at the start of
a recording, then ``predict(X)`` once per 5 s window, in time order. The
ladder is built so that each rung isolates one source of information:

* ``ConstantModel``        - no information at all (the published floor).
* ``StatelessRidge``       - the EEG window only (what EEGNet and CTNet see).
* ``ElapsedQuantile``      - the elapsed time only, no EEG whatsoever.
* ``ElapsedFeatRidge``     - elapsed time plus the current window's EEG.
* ``ReservoirStream``      - elapsed time plus the EEG history, integrated by
  a recurrent state. With ``holographic=True`` the recurrent matrix is the
  Holographic-Reservoir one (expander graph with SHA-256 derived entropy on
  the diagonal); with ``holographic=False`` it is the same construction
  without that entropy, which is a plain echo-state network.

The split between ``ElapsedQuantile`` and the rungs above it is the one that
matters: it separates the free lunch of knowing how long the recording has
been running from any real contribution of the EEG signal.
"""

import numpy as np

CAP_S = 600.0
WINDOW_S = 5.0
BIN_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
BIN_WEIGHTS = (10.0, 5.0, 3.0, 1.0)


# --------------------------------------------------------------------------
# metric and sample weights
# --------------------------------------------------------------------------
def bin_index(y):
    """Index of the metric's severity range each latency falls in."""
    return np.clip(
        np.digitize(np.asarray(y, float), BIN_EDGES[1:-1], right=False),
        0, len(BIN_WEIGHTS) - 1)


def wbmae(y_true, y_pred):
    """W-bMAE of one recording: weighted mean of the non-empty ranges."""
    y_true = np.asarray(y_true, float)
    err = np.abs(np.asarray(y_pred, float) - y_true)
    idx = bin_index(y_true)
    num = den = 0.0
    for b, weight in enumerate(BIN_WEIGHTS):
        mask = idx == b
        if mask.any():
            num += weight * err[mask].mean()
            den += weight
    return float(num / den) if den else float("nan")


def wbmae_recording_mean(y_true, y_pred, record_id):
    """The ranking metric: W-bMAE per recording, averaged over recordings."""
    record_id = np.asarray(record_id)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    scores = [wbmae(y_true[record_id == r], y_pred[record_id == r])
              for r in np.unique(record_id)]
    return float(np.mean(scores))


def metric_weights(y, record_id):
    """Per-window weights whose weighted MAE equals the ranking metric.

    Inside a recording the metric averages the non-empty ranges, so a window
    counts as ``w_bin / (n_windows_in_its_bin * sum_of_present_weights)``;
    recordings then count equally, hence the 1/n_recordings factor. Fitting
    against these weights optimises the metric itself rather than a plain
    mean error, which would be dominated by the long early stretch of every
    recording.
    """
    y = np.asarray(y, float)
    record_id = np.asarray(record_id)
    weights = np.zeros(len(y))
    records = np.unique(record_id)
    idx = bin_index(y)
    for r in records:
        in_rec = record_id == r
        present = [b for b in range(len(BIN_WEIGHTS))
                   if (in_rec & (idx == b)).any()]
        den = sum(BIN_WEIGHTS[b] for b in present)
        for b in present:
            mask = in_rec & (idx == b)
            weights[mask] = BIN_WEIGHTS[b] / (mask.sum() * den)
    return weights / len(records)


def weighted_quantile(values, weights, q):
    """Weighted q-quantile, used for the metric-weighted median."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    if cum[-1] <= 0:
        return float(np.median(values))
    return float(np.interp(q * cum[-1], cum, values))


# --------------------------------------------------------------------------
# elapsed-time encoding
# --------------------------------------------------------------------------
def elapsed_features(elapsed):
    """Encode elapsed seconds; saturating terms so late windows stay apart."""
    e = np.asarray(elapsed, float).reshape(-1, 1) / CAP_S
    return np.concatenate([
        np.ones_like(e),
        np.minimum(e, 4.0),
        np.sqrt(np.minimum(e, 16.0)),
        np.log1p(np.minimum(e, 16.0)),
        np.exp(-np.minimum(e, 16.0)),
    ], axis=1)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class ConstantModel:
    """Always the same latency - the published floor (constant 35 s)."""

    def __init__(self, value=35.0):
        self.value = float(value)

    def fit(self, data):
        self.value = weighted_quantile(
            data["y"], metric_weights(data["y"], data["record_id"]), 0.5)
        return self

    def reset_state(self):
        pass

    def predict_feats(self, feats, elapsed=None, record_id=None):
        return np.full(len(feats), np.clip(self.value, 0.0, CAP_S))

    def predict(self, X):
        return np.full(len(X), np.clip(self.value, 0.0, CAP_S))


class _RidgeCore:
    """Weighted linear readout fitted to the metric's own loss.

    The ranking metric is a weighted *absolute* error, so a plain ridge -
    which minimises the squared error - is fitted to the wrong thing and
    gets dragged around by the long tail of early windows. The closed-form
    weighted ridge is therefore only the starting point: ``l1_iters`` rounds
    of iteratively reweighted least squares (each sample reweighted by
    ``1 / max(|residual|, delta)``) converge towards the weighted L1
    solution while keeping every step a single linear solve.

    Set ``l1_iters=0`` to get the plain ridge back.
    """

    def __init__(self, alpha=10.0, l1_iters=12):
        self.alpha = alpha
        self.l1_iters = l1_iters
        self.mean = self.scale = self.coef = None

    def _design(self, Z):
        return np.concatenate([(Z - self.mean) / self.scale,
                               np.ones((len(Z), 1))], axis=1)

    def _solve(self, Zs, y, weights):
        sw = np.sqrt(np.maximum(weights, 0.0) * len(weights))[:, None]
        A = (Zs * sw).T @ (Zs * sw) + self.alpha * np.eye(Zs.shape[1])
        b = (Zs * sw).T @ (y[:, None] * sw)
        return np.linalg.solve(A, b).ravel()

    def fit(self, Z, y, weights):
        Z = np.asarray(Z, float)
        y = np.asarray(y, float)
        weights = np.asarray(weights, float)
        self.mean = Z.mean(axis=0)
        self.scale = Z.std(axis=0) + 1e-8
        Zs = self._design(Z)

        self.coef = self._solve(Zs, y, weights)
        for _ in range(self.l1_iters):
            residual = np.abs(Zs @ self.coef - y)
            # One window (5 s) as the floor: below it the error no longer
            # matters for the metric and dividing by it would explode.
            delta = max(WINDOW_S, 0.1 * np.median(residual))
            self.coef = self._solve(Zs, y, weights / np.maximum(residual, delta))
        return self

    def predict(self, Z):
        return np.clip(self._design(np.asarray(Z, float)) @ self.coef,
                       0.0, CAP_S)


class StatelessRidge:
    """Ridge on the current window's EEG features - no memory of the past.

    This is the information a per-window convolutional decoder such as
    EEGNet has access to, in the cheapest form.
    """

    def __init__(self, alpha=10.0):
        self.core = _RidgeCore(alpha)

    def fit(self, data):
        self.core.fit(data["feats"], data["y"],
                      metric_weights(data["y"], data["record_id"]))
        return self

    def reset_state(self):
        pass

    def predict_feats(self, feats, elapsed=None, record_id=None):
        return self.core.predict(feats)

    def predict(self, X):
        from features import window_features
        return self.predict_feats(window_features(X))


class ElapsedQuantile:
    """Metric-weighted conditional median of the latency given elapsed time.

    Uses no EEG at all: it only counts how many windows have gone by. This is
    the control that says how much of any improvement is simply the
    arithmetic of a recording that is still running.
    """

    def __init__(self, n_bins=40, smooth=3):
        self.n_bins = n_bins
        self.smooth = smooth
        self.edges = None
        self.table = None
        self.k = 0

    def fit(self, data):
        elapsed = np.asarray(data["elapsed"], float)
        y = np.asarray(data["y"], float)
        weights = metric_weights(y, data["record_id"])
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges = np.unique(np.quantile(elapsed, quantiles))
        idx = np.digitize(elapsed, self.edges)
        raw = np.full(len(self.edges) + 1, np.nan)
        for b in range(len(raw)):
            mask = idx == b
            if mask.any() and weights[mask].sum() > 0:
                raw[b] = weighted_quantile(y[mask], weights[mask], 0.5)
        # Fill empty cells, then smooth: the table must stay usable at
        # elapsed times no training recording reached.
        valid = ~np.isnan(raw)
        raw = np.interp(np.arange(len(raw)), np.flatnonzero(valid), raw[valid])
        if self.smooth > 1:
            pad = self.smooth // 2
            kernel = np.ones(self.smooth) / self.smooth
            raw = np.convolve(np.pad(raw, (pad, pad), mode="edge"),
                              kernel, mode="valid")
        self.table = raw
        return self

    def lookup(self, elapsed):
        idx = np.digitize(np.asarray(elapsed, float), self.edges)
        return np.clip(self.table[idx], 0.0, CAP_S)

    def reset_state(self):
        self.k = 0

    def predict_feats(self, feats, elapsed, record_id=None):
        return self.lookup(elapsed)

    def predict(self, X):
        out = self.lookup([self.k * WINDOW_S] * len(X))
        self.k += len(X)
        return out


class ElapsedFeatRidge:
    """Ridge on the current window's EEG features *and* the elapsed time.

    Stateless in the EEG - it never looks back - but it knows the clock.
    """

    def __init__(self, alpha=10.0, prior=None):
        self.core = _RidgeCore(alpha)
        self.prior = prior
        self.k = 0

    def _inputs(self, feats, elapsed):
        parts = [np.asarray(feats, float), elapsed_features(elapsed)]
        if self.prior is not None:
            parts.append(self.prior.lookup(elapsed).reshape(-1, 1) / CAP_S)
        return np.concatenate(parts, axis=1)

    def fit(self, data):
        if self.prior is None:
            self.prior = ElapsedQuantile().fit(data)
        Z = self._inputs(data["feats"], data["elapsed"])
        self.core.fit(Z, data["y"],
                      metric_weights(data["y"], data["record_id"]))
        return self

    def predict_feats(self, feats, elapsed, record_id=None):
        return self.core.predict(self._inputs(feats, elapsed))

    def reset_state(self):
        self.k = 0

    def predict(self, X):
        from features import window_features
        feats = window_features(X)
        out = self.predict_feats(feats, [self.k * WINDOW_S] * len(feats))
        self.k += len(feats)
        return out


class ReservoirStream:
    """Holographic reservoir over the EEG history of the current recording.

    The recurrent state is carried from window to window inside a recording
    and cleared by ``reset_state()``, which is what the streamed evaluation
    gives a model and what a per-window decoder cannot use. The readout sees
    the state, the elapsed-time encoding and the elapsed-time prior.

    ``holographic=True`` builds the recurrent matrix exactly as
    ``holographic_reservoir.HolographicReservoirModel`` does, including the
    SHA-256 derived entropy added to its diagonal; ``holographic=False``
    skips only that entropy term and is therefore a plain echo-state
    network, which is how the holographic part itself gets measured.
    """

    def __init__(self, size=300, spectral_radius=0.9, leak=0.3, alpha=10.0,
                 seed=0, holographic=True, input_scale=0.5, prior=None,
                 use_prior=True, use_feats=True):
        self.size = size
        self.spectral_radius = spectral_radius
        self.leak = leak
        self.alpha = alpha
        self.seed = seed
        self.holographic = holographic
        self.input_scale = input_scale
        self.prior = prior
        self.use_prior = use_prior
        self.use_feats = use_feats
        self.core = _RidgeCore(alpha)
        self.mean_in = self.scale_in = None
        self.W_in = self.W = None
        self.state = None
        self.k = 0

    # -- reservoir construction (mirrors HolographicReservoirModel) -------
    def _build(self, input_dim):
        rng = np.random.default_rng(self.seed)
        self.W_in = rng.uniform(-self.input_scale, self.input_scale,
                                size=(self.size, input_dim))
        density = 0.1
        W = rng.standard_normal((self.size, self.size))
        W = W * (rng.random((self.size, self.size)) < density)
        if self.holographic:
            # The Holographic-Reservoir entropy term: deterministic SHA-256
            # frames from the simulated-ASIC backend, added to the diagonal.
            from holographic_reservoir.backends import SimulatedASICBackend
            backend = SimulatedASICBackend(random_seed=self.seed)
            frames = backend.mine(b"init", cycles=max(1, self.size // 8))
            entropy = np.concatenate(
                [np.frombuffer(f, dtype=np.uint8) for f in frames]
            ).astype(np.float64)
            entropy = (entropy[:self.size] / 127.5 - 1.0) * 0.05
            np.fill_diagonal(W, np.diag(W) + entropy)
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W = W * (self.spectral_radius / radius)
        self.W = W

    def _states(self, feats, record_id):
        """Reservoir states, restarted at every recording boundary."""
        u = (np.asarray(feats, float) - self.mean_in) / self.scale_in
        record_id = np.asarray(record_id)
        states = np.zeros((len(u), self.size))
        x = np.zeros(self.size)
        previous = None
        for t in range(len(u)):
            if previous is None or record_id[t] != previous:
                x = np.zeros(self.size)
                previous = record_id[t]
            x = (1.0 - self.leak) * x + self.leak * np.tanh(
                self.W_in @ u[t] + self.W @ x)
            states[t] = x
        return states

    def _inputs(self, states, elapsed, feats=None):
        """Readout inputs: the state, the current window, and the clock.

        The current window's features go in alongside the state on purpose.
        With a leak of 0.3 the state is a blurred average of the recent
        past, so a readout given only the state cannot see the window it is
        predicting for, and the model would not even contain the stateless
        one it is meant to be compared against. Including both makes this
        model a strict superset of ``ElapsedFeatRidge``, so the difference
        between them measures what the EEG *history* adds - which is the
        question - rather than what the state throws away.
        """
        parts = [states]
        if feats is not None and self.use_feats:
            parts.append(np.asarray(feats, float))
        parts.append(elapsed_features(elapsed))
        if self.prior is not None and self.use_prior:
            parts.append(self.prior.lookup(elapsed).reshape(-1, 1) / CAP_S)
        return np.concatenate(parts, axis=1)

    def fit(self, data):
        feats = np.asarray(data["feats"], float)
        self.mean_in = feats.mean(axis=0)
        self.scale_in = feats.std(axis=0) + 1e-8
        self._build(feats.shape[1])
        if self.prior is None and self.use_prior:
            self.prior = ElapsedQuantile().fit(data)
        states = self._states(feats, data["record_id"])
        Z = self._inputs(states, data["elapsed"], feats)
        self.core.fit(Z, data["y"],
                      metric_weights(data["y"], data["record_id"]))
        return self

    def predict_feats(self, feats, elapsed, record_id):
        states = self._states(feats, record_id)
        return self.core.predict(self._inputs(states, elapsed, feats))

    # -- streaming contract ----------------------------------------------
    def reset_state(self):
        self.state = np.zeros(self.size)
        self.k = 0

    def predict(self, X):
        from features import window_features
        if self.state is None:
            self.reset_state()
        feats = window_features(X)
        u = (feats - self.mean_in) / self.scale_in
        out = np.empty(len(u))
        for i in range(len(u)):
            self.state = (1.0 - self.leak) * self.state + self.leak * np.tanh(
                self.W_in @ u[i] + self.W @ self.state)
            Z = self._inputs(self.state[None], [self.k * WINDOW_S],
                             feats[i:i + 1])
            out[i] = self.core.predict(Z)[0]
            self.k += 1
        return out


# ---------------------------------------------------------------------------
# explicit-memory and residual variants (added after the sweep, iteration 3)
# ---------------------------------------------------------------------------
def ema_features(feats, record_id, halflives=(2, 6, 24)):
    """Causal exponential moving averages, restarted at each recording.

    A half-life of 2, 6 and 24 windows is 10 s, 30 s and 2 min of history.
    """
    feats = np.asarray(feats, float)
    record_id = np.asarray(record_id)
    out = []
    for halflife in halflives:
        decay = 0.5 ** (1.0 / halflife)
        running = np.zeros((len(feats), feats.shape[1]))
        state = np.zeros(feats.shape[1])
        previous = None
        for t in range(len(feats)):
            if previous is None or record_id[t] != previous:
                state = feats[t].copy()
                previous = record_id[t]
            state = decay * state + (1.0 - decay) * feats[t]
            running[t] = state
        out.append(running)
    return np.concatenate(out, axis=1)


class EmaRidge:
    """Window, clock and explicit moving averages of the features.

    The memory is one exponential moving average per timescale, carried
    across the windows of a recording and cleared by ``reset_state()``. It
    is the simplest form of memory that fits the streamed contract, and the
    sweep found it beats the recurrent reservoir on this task.
    """

    def __init__(self, alpha=10.0, halflives=(2, 6, 24)):
        self.core = _RidgeCore(alpha)
        self.halflives = halflives
        self.prior = None
        self.ema = None
        self.k = 0

    def _inputs(self, feats, elapsed, record_id):
        ema = ema_features(feats, record_id, self.halflives)
        parts = [np.asarray(feats, float), ema, elapsed_features(elapsed)]
        if self.prior is not None:
            parts.append(self.prior.lookup(elapsed).reshape(-1, 1) / CAP_S)
        return np.concatenate(parts, axis=1)

    def fit(self, data):
        self.prior = ElapsedQuantile().fit(data)
        Z = self._inputs(data["feats"], data["elapsed"], data["record_id"])
        self.core.fit(Z, data["y"],
                      metric_weights(data["y"], data["record_id"]))
        return self

    def predict_feats(self, feats, elapsed, record_id):
        return self.core.predict(self._inputs(feats, elapsed, record_id))

    # -- streaming contract ----------------------------------------------
    def reset_state(self):
        self.ema = None
        self.k = 0

    def predict(self, X):
        """One window at a time, carrying the averages across the recording."""
        from features import window_features
        feats = window_features(X)
        if getattr(self, "ema", None) is None:
            self.reset_state()
        out = np.empty(len(feats))
        for i in range(len(feats)):
            row = feats[i]
            if self.ema is None:
                self.ema = [row.copy() for _ in self.halflives]
            for j, halflife in enumerate(self.halflives):
                decay = 0.5 ** (1.0 / halflife)
                self.ema[j] = decay * self.ema[j] + (1.0 - decay) * row
            parts = [row[None], np.concatenate(self.ema)[None],
                     elapsed_features([self.k * WINDOW_S])]
            if self.prior is not None:
                parts.append(self.prior.lookup(
                    [self.k * WINDOW_S]).reshape(-1, 1) / CAP_S)
            out[i] = self.core.predict(np.concatenate(parts, axis=1))[0]
            self.k += 1
        return out


class ReservoirResidual:
    """Stateless model first; the reservoir state only corrects it.

    The base model is fitted and frozen, then a strongly regularised readout
    maps the reservoir state onto the base model's residual. The history can
    only add to the prediction, so a tie here means the state genuinely
    carries nothing the current window does not already say.
    """

    def __init__(self, base_alpha=10.0, size=200, leak=0.5, alpha=100.0,
                 seed=0):
        self.base = ElapsedFeatRidge(alpha=base_alpha)
        self.res = ReservoirStream(size=size, leak=leak, alpha=alpha,
                                   seed=seed)
        self.correction = _RidgeCore(alpha)

    def fit(self, data):
        self.base.fit(data)
        base_pred = self.base.predict_feats(data["feats"], data["elapsed"])
        residual = np.asarray(data["y"], float) - base_pred

        feats = np.asarray(data["feats"], float)
        self.res.mean_in = feats.mean(axis=0)
        self.res.scale_in = feats.std(axis=0) + 1e-8
        self.res._build(feats.shape[1])
        states = self.res._states(feats, data["record_id"])
        self.correction.fit(states, residual,
                            metric_weights(data["y"], data["record_id"]))
        return self

    def predict_feats(self, feats, elapsed, record_id):
        base_pred = self.base.predict_feats(feats, elapsed)
        states = self.res._states(feats, record_id)
        delta = self.correction.predict(states)
        # The correction's own readout clips to [0, CAP]; recentre it so it
        # is a signed correction rather than a prediction in its own right.
        delta = delta - self.correction.coef[-1]
        return np.clip(base_pred + delta, 0.0, CAP_S)
