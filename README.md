# Sleep-Onset Latency from Four-Channel Home EEG (Track 03)

Entry for **Track 03 — Sleep onset** of the [EEG/EMG Foundation Challenge
2026](https://neural-interfaces26.github.io/) (NeuralBench, Brain & Body Workshop at
NeurIPS 2026). From four-channel at-home Muse EEG, predict how many seconds remain until
the sleeper's first N2 epoch, capped at 600 s. Ranking metric: **W-bMAE per recording,
averaged over recordings** (lower is better).

Author: Francisco Angulo de Lafuente ([Agnuxo](https://github.com/Agnuxo1)), Madrid.
Licence: Apache-2.0.

---

## Headline result

**W-bMAE 55.816 on the public test split, scored through the competition's own evaluation
harness — 4.3 s better than the best published baseline.**

| System | W-bMAE (s) |
|---|---|
| **This entry** | **55.82** |
| CTNet (published by the organisers) | 60.10 |
| EEGNet (published by the organisers) | 60.60 |
| Constant 35 s (published by the organisers) | 60.70 |
| REVE, fine-tuned (published by the organisers) | 72.90 |

That number is not self-reported from my own scoring code: `code/verify_official.py` loads
the built submission exactly as the platform does and scores it with the kit's own
`predict_streams` and `binned_mae` over the 40 public test recordings. It reproduces the
offline figure to the third decimal.

Under five-fold cross-validation on the training release the same system reaches **51.21 ±
1.44** with participants held out entirely, and **50.06 ± 2.05** on held-out recordings of
participants seen in training — roughly 10 s below the published floor in both regimes.

It is also cheap to run: the submission is **7.6 kB**, imports nothing but numpy, and
scores the whole public test split in **11.9 s** against a limit of 80 minutes on one
A100. Training used **CPU only** — no GPU at any point, for any model in this repository.

![Model ladder](figures/escalera.png)

## How it works

Three ingredients, in order of how much each is worth.

**1. Fitting the readout to the competition's actual loss (−5.8 s).** The ranking metric is
a *weighted absolute* error; a ridge minimises a squared one. Each window is given the
weight that makes a weighted mean absolute error equal the ranking metric exactly, and the
linear readout is then fitted by iteratively reweighted least squares. This single change
is worth more than any architecture tried here.

**2. Classical spectral features (−7.5 s against the constant).** Per 5 s window and
channel: log and relative band powers in delta, theta, alpha, sigma and beta; four log
ratios that track drowsiness; Hjorth activity, mobility and complexity; and a clipping
flag, since the recordings are raw and one blink otherwise dominates every power.

**3. The clock the streamed harness makes available (−2.7 s more).** The target is
`y(t) = min(600, D − t)`, with `t` the time elapsed since the recording started and `D` the
first N2 epoch. The official harness
([`benchmark_utils/streaming.py`](https://github.com/neural-interfaces26/2026-competition))
feeds each recording's windows one at a time, in time order, from the start of the
recording, deep-copies the model per recording and calls `reset_state()` first — its own
docstring says *"A model can adapt within a stream."* So a model that keeps state knows `t`
by counting its own calls: causal, nothing from the future. A per-window decoder cannot.

![Why a stateless model cannot track the target](figures/perfil_temporal.png)

A short memory on top of that — one exponential moving average of the features with a 10 s
half-life, cleared between recordings — adds a further 0.5 s and wins in all ten
cross-validation folds of both regimes.

Nothing here touches the sealed split, uses non-permitted data or modifies the scoring
harness (rule 06). It uses the state the harness explicitly offers.

## Results in full

Five-fold cross-validation, two regimes, three seeds for every model with a random
substrate. `seen` holds out recordings of participants kept in training; `unseen` holds out
whole participants.

| Model | seen | unseen |
|---|---|---|
| **I explicit 10 s memory (submitted)** | **50.06 ± 2.05** | **51.21 ± 1.44** |
| D clock + window | 50.41 ± 2.09 | 51.73 ± 1.57 |
| J reservoir as a correction | 50.77 ± 1.99 | 52.11 ± 1.47 |
| F plain ESN | 51.47 ± 1.82 | 52.71 ± 1.37 |
| E holographic reservoir | 51.47 ± 1.82 | 52.72 ± 1.37 |
| H reservoir, state only | 52.75 ± 2.43 | 53.84 ± 1.37 |
| G reservoir, no clock | 53.69 ± 1.08 | 54.55 ± 1.40 |
| B stateless ridge | 53.69 ± 1.42 | 54.40 ± 1.01 |
| C clock only | 55.76 ± 2.98 | 56.75 ± 1.56 |
| A constant | 61.89 ± 1.29 | 61.88 ± 1.02 |

Fold-to-fold variation is ±1.5–2.5 s, larger than the gaps between the leading models, so
the comparisons that decide anything are the ones made *within* each fold, where the
held-out recordings are the same for both models (negative favours the first):

| Comparison | seen | unseen | folds won |
|---|---|---|---|
| B − A — adding the EEG features | −8.20 ± 1.34 | −7.48 ± 0.69 | 5 of 5 |
| C − A — the clock on its own | −6.12 ± 2.45 | −5.13 ± 2.22 | 5 of 5 |
| D − B — adding the clock | −3.28 ± 2.55 | −2.67 ± 1.56 | 5 of 5 |
| I − D — adding the 10 s memory | −0.35 ± 0.16 | −0.52 ± 0.24 | 5 of 5, 5 of 5 |
| H − E — removing the current window | +1.28 ± 0.67 | +1.12 ± 0.28 | 0 of 5 |
| E − D — the reservoir against the control | +1.06 ± 0.77 | +0.99 ± 1.02 | 1 of 5 |
| E − F — the SHA-256 entropy term | +0.00 ± 0.00 | +0.00 ± 0.00 | 0 of 5 |

## Reproducing

The public data (NEMAR `nm000287`, 540 recordings, ~1.1 GB) is **not** redistributed here:
it is downloaded from NEMAR under its own terms, and no signal, feature cache or
participant-level artefact from it is committed to this repository.

```bash
pip install -r requirements.txt
python code/prepare_data.py        # downloads NEMAR nm000287 (~1.1 GB)
python code/extract_cache.py       # per-window spectral features -> cache/
python code/run_cv.py --folds 5 --seeds 0 1 2
python code/build_submission.py --out submissions/memoria_explicita
python code/verify_official.py --submission submissions/memoria_explicita
```

`code/test_ladder.py` checks the parts that are easy to get silently wrong: that the sample
weights reproduce the ranking metric exactly, and that predicting one window at a time —
with a deep copy and `reset_state()` per recording, as the platform does — matches the
offline fit to 1.4e-14, for the submitted model and for the reservoir.

### Environment notes

- `code/kit_compat.py` works around version skews between the starting kit (branch `main`)
  and the `neuralbench==0.4.0` the kit itself pins, plus `os.pwrite`, which does not exist
  on Windows. None of the patches changes any data or split.
- Window extraction needs `NEURALBENCH_EXTRACT_CPUS=1` on a machine with a modest pagefile:
  the default spawns one torch-loading worker per core.

---

## Observations

Things that did not work, or that qualify the result above. None of them changes the
headline, and all of them are reported because leaving them out would be picking the data.

**The Holographic-Reservoir did not contribute.** The author's
[Holographic-Reservoir](https://pypi.org/project/holographic-reservoir/) — a sparse
expander-graph reservoir with deterministic SHA-256 entropy on its diagonal, read out
linearly — was the intended core of this entry, and on paper it fits a streamed task well.
Measured against the same control under the same validation and compute, it came out 1.0 s
*worse* and lost in four folds of five in both regimes. As a correction on top of the
control it still lost by 0.4 s. The sweep was blunter: the shorter its memory, the better
it scored — at a leak of 0.1 it reached 80–112 s, at 0.7 it merely tied. The likely cause
is that its state dimensions are strongly autocorrelated and ill-condition the fit; it
degrades the readout even on training data, so this is not overfitting. It is kept in the
repository, implemented and tuned, because the measurement is worth having.

**The SHA-256 entropy term is not measurable.** E − F is `+0.00 ± 0.00` in both regimes:
adding that entropy to the recurrent matrix's diagonal gives the same score to the last
decimal as leaving it out. What remains is an echo-state network.

**One number points the other way.** On the public test split the reservoir scores 55.741,
0.08 s *better* than the submitted model — the opposite of the cross-validation ordering.
With 40 recordings that difference is noise, but it is there.

**The clock assumes the sealed streams start where the public ones do.** If the sealed
cohort's windows were trimmed differently, the elapsed-time counter would be biased and
part of the advantage would go; model G is the variant that does not depend on it.

**Scope.** Everything is validated on the public Muse split; the sealed cohort is private.
The feature front end is deliberately classical and no claim is made that it is the best
possible encoder. No leaderboard position is claimed: nothing has been submitted at the
time of writing.
