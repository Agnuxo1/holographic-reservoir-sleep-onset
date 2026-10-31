# Method description — Track 03 (Sleep onset)

Required by rules 04 (declare every external corpus and a compute estimate), 07
(reproducibility audit) and 08 (method description, training and inference code,
pre-training disclosures).

## 0. Result

W-bMAE **55.816** on the public test split, scored through the competition's own harness
(`predict_streams` + `binned_mae` over the 40 public test recordings), against 60.10 for
CTNet, 60.60 for EEGNet and 60.70 for the constant baseline published by the organisers.
Five-fold cross-validation on the training release: 51.21 ± 1.44 with participants held
out, 50.06 ± 2.05 on held-out recordings of participants seen in training.

Submission size 7.6 kB, numpy-only inference, 11.9 s for the 5,869 public test windows
against an 80-minute limit. Trained on CPU alone.

**Participant:** Francisco Angulo de Lafuente (Agnuxo), Madrid, Spain. Solo participant,
one-person team, no institutional affiliation.

## 1. Pre-training disclosure

**No pre-training of any kind. No external corpus. No pre-trained weights.**

The model is fitted from scratch on the Track 03 public training release only
(`Interaxon2026Muse`, NEMAR `nm000287`, 412 training recordings / 61,303 windows as
staged by the official `eeg/_sleep_onset_stream` task). No foundation model, no transfer
from another EEG dataset, no self-supervised stage, no synthetic augmentation.

## 2. Model

A linear readout over three groups of inputs, computed per 5 s window of the four Muse
channels (TP9, AF7, AF8, TP10) at 128 Hz, unfiltered, in microvolts.

**Features (72 per window).** Amplitudes clipped to ±500 µV first, since the recordings
are raw and a single blink otherwise dominates every power in the window; then the window
is mean-centred. A Welch periodogram (256-sample Hann segments, 50% overlap) gives, per
channel: log band powers and relative band powers in delta (0.5–4 Hz), theta (4–8),
alpha (8–12), sigma (12–16) and beta (16–30); four log ratios (theta/alpha, delta/alpha,
sigma/alpha, (delta+theta)/(alpha+sigma)); and Hjorth activity, mobility and complexity
plus the fraction of clipped samples.

**Memory (72 per window).** One causal exponential moving average of those features, with
a half-life of two windows (10 s), restarted at the beginning of each recording.

**Clock (6 per window).** The time elapsed since the recording started, encoded as
`1, min(e,4), sqrt(min(e,16)), log1p(min(e,16)), exp(-min(e,16))` with `e` in units of
600 s, plus one metric-weighted conditional median of the latency given the elapsed time,
tabulated on the training split in 40 quantile bins and smoothed over three bins.

The elapsed time is obtained by counting the model's own `predict` calls since
`reset_state()`. This is causal and uses no information from the future: the harness
delivers each recording's windows one at a time, in time order, from the start of the
recording, deep-copies the model per recording and calls `reset_state()` first
(`benchmark_utils/streaming.py`, whose docstring states that a model may adapt within a
stream). Nothing in the sealed split is touched and the scoring harness is unmodified.

**Readout.** One linear map from the 150 inputs (standardised, with an intercept) to the
latency in seconds, clipped to [0, 600].

**Fitting the readout to the competition's loss.** The ranking metric is a weighted
*absolute* error, so each window is given the weight that makes a weighted mean absolute
error equal the ranking metric exactly: `w_bin / (n_windows_in_that_bin_of_that_recording
× sum_of_present_bin_weights) / n_recordings`. A closed-form weighted ridge
(α = 10) initialises the solution, then twelve rounds of iteratively reweighted least
squares (each sample reweighted by `1 / max(|residual|, 5 s)`) converge towards the
weighted L1 solution. On the validation split this is worth 5.8 s of W-bMAE over the
plain squared-error fit (58.71 → 52.95), more than any of the memory mechanisms tried.

## 3. Validation

Chosen on the release's own validation split (88 recordings of participants with no test
recording), never on the 40 test recordings. Also cross-validated five-fold in two
regimes — grouped by participant (no recording of a held-out participant in training) and
held-out recordings of participants kept in training — because the sealed phase scores a
macro-average over participants seen and unseen in training.

## 4. Compute estimate

Training and all experiments ran on **CPU only**, on one desktop machine: no GPU was used
at any point, for this model or for any of the controls and ablations. One fit of the
submitted model over the 61,303 training windows takes about 20 s (feature extraction of
the whole public release, done once and cached, takes about 2 min). The full experiment
programme — ten model variants, two validation regimes, five folds, three seeds, plus two
hyperparameter sweeps and a memory probe — is on the order of one CPU-hour in total.

Inference: the submitted model is numpy only, about 1 ms per window; the 5,869 windows of
the public test split are scored in well under a minute, far inside the 80-minute limit.
Submission size: 7.6 kB.

## 5. Code

Training and inference code, the experiment log and the reproduction instructions are
public at https://github.com/Agnuxo1/holographic-reservoir-sleep-onset (Apache-2.0). The
public data is not redistributed there: it is downloaded from NEMAR under its own terms,
and no signal or participant-level artefact is committed.

## 6. Observations

Reported because they are part of the method, not because they change the result.

The model above is the *control*. The author's own Holographic-Reservoir — a sparse
expander-graph reservoir with SHA-256 derived entropy on its diagonal, read out
linearly — was the intended core of the entry and was measured against it under the same
validation and the same compute. It did not help: at best it tied, and the sweep showed
that the less memory it kept, the better it scored. The reservoir with and without its
SHA-256 entropy term scored the same to within 0.02 s across three seeds. The full
ablation ladder and the numbers are in the public repository, including the runs that went
against the hypothesis.
