# Holographic Reservoir for Sleep-Onset Latency (Track 03)

Entry for **Track 03 — Sleep onset** of the [EEG/EMG Foundation Challenge
2026](https://neural-interfaces26.github.io/) (NeuralBench, Brain & Body Workshop at
NeurIPS 2026). The task: from four-channel at-home Muse EEG, predict how many seconds
remain until the sleeper's first N2 epoch, capped at 600 s. Ranking metric: **W-bMAE per
recording, averaged over recordings** (lower is better).

Author: Francisco Angulo de Lafuente ([Agnuxo](https://github.com/Agnuxo1)), Madrid.
Licence: Apache-2.0.

---

## The observation this is built on

The organisers publish these reference scores on the public Muse split:

| Published baseline | W-bMAE (s) |
|---|---|
| Constant 35 s (validation-selected) | 60.7 |
| EEGNet | 60.6 |
| CTNet | 60.1 |
| REVE, fine-tuned | 72.9 |

Three trained decoders, two of them modern architectures, and none of them beats a
constant by more than a rounding error. One of them is clearly worse. When a reference
table collapses onto its own floor like that, the usual cause is not a shortage of model
capacity — something the models are not being given, or not being fitted to, is doing the
work instead.

Two candidates were tested here, and the smaller-sounding one turned out to matter more.

**The clock.** The target is

```
y(t) = min(600, D - t)
```

where `t` is the time elapsed since the recording started and `D` the moment of the first
N2 epoch. The official evaluation harness
([`benchmark_utils/streaming.py`](https://github.com/neural-interfaces26/2026-competition))
feeds a recording's windows **one at a time, in time order, from the start of the
recording**, deep-copies the model for each recording and calls `reset_state()` first. Its
own docstring says: *"A model can adapt within a stream."* So a model that keeps state
knows `t` exactly — it only has to count its own calls, which is causal and uses nothing
from the future. A per-window decoder cannot know `t` no matter how well it reads the EEG.
Measured contribution: **−2.7 s** on top of the window features, and −5.1 s on its own
against the constant. Real, but not the whole story — the task's durations vary enough,
and the 600 s cap flattens enough early windows, that the clock alone correlates with the
target at only 0.42.

**The loss.** The ranking metric is a *weighted absolute* error, while a ridge minimises a
squared one. Fitting the same linear readout to the competition's own weighting, by
iteratively reweighted least squares, is worth **−5.8 s** of W-bMAE on the validation
split — more than the clock, and more than any memory mechanism tried here. It is the
single largest effect measured in this work, and it is not an architecture.

Nothing here touches the sealed split, uses non-permitted data or modifies the scoring
harness (rule 06). It uses the state the harness explicitly offers.

## Where the technology comes from

The intended core of this entry was the author's
[Holographic-Reservoir](https://pypi.org/project/holographic-reservoir/) (Apache-2.0): a
sparse expander-graph reservoir scaled to a target spectral radius, with deterministic
SHA-256 derived entropy added to its diagonal, read out linearly. Inside a recording the
state integrates the EEG history; `reset_state()` clears it between recordings, exactly as
the harness requires — on paper a natural fit for a streamed task.

It did not work. It is kept in this repository, fully implemented, tuned and measured,
because the measurement is the point: the results section reports exactly how much it
contributed, which is nothing, and the submitted model does not use it. The ladder below
was built so that this question could have only one answer, whichever way it came out.

## What is actually measured

Knowing the clock is free arithmetic, and it would be easy — and wrong — to bank that gain
as a win for the reservoir. So every rung of the ladder is run separately, under the same
validation and the same compute, and the contributions are reported apart:

| Model | Clock | Current window's EEG | EEG history | What it isolates |
|---|---|---|---|---|
| A constant | no | no | no | the published floor |
| B stateless ridge | no | yes | no | what EEGNet can see |
| C clock only | yes | **no** | no | the arithmetic alone |
| D clock + window | yes | yes | no | the instantaneous EEG's contribution |
| E holographic reservoir | yes | yes | yes | the proposed system |
| F plain ESN | yes | yes | yes | E without the SHA-256 entropy |
| G reservoir, no clock | state only | yes | yes | the reservoir on its own |

**E − D** is the reservoir's real contribution. **E − F** is the contribution of the
holographic part specifically. Both are reported whatever their sign.

## Results

Five-fold cross-validation, two regimes, three seeds for every model with a random
substrate. Fold-to-fold variation is ±1.5–2.5 s, larger than the gaps between the leading
models, so the table is followed by the within-fold comparisons, which are what actually
decide anything.

![Model ladder](figures/escalera.png)

| Model | seen | unseen |
|---|---|---|
| A constant | 61.89 ± 1.29 | 61.88 ± 1.02 |
| C clock only | 55.76 ± 2.98 | 56.75 ± 1.56 |
| B stateless ridge | 53.69 ± 1.42 | 54.40 ± 1.01 |
| G reservoir, no clock | 53.69 ± 1.08 | 54.55 ± 1.40 |
| H reservoir, state only | 52.75 ± 2.43 | 53.84 ± 1.37 |
| E holographic reservoir | 51.47 ± 1.82 | 52.72 ± 1.37 |
| F plain ESN | 51.47 ± 1.82 | 52.71 ± 1.37 |
| J reservoir as a correction | 50.77 ± 1.99 | 52.11 ± 1.47 |
| D clock + window | 50.41 ± 2.09 | 51.73 ± 1.57 |
| **I explicit 10 s memory (submitted)** | **50.06 ± 2.05** | **51.21 ± 1.44** |

Paired within each fold (negative favours the first model):

| Comparison | seen | unseen | folds won |
|---|---|---|---|
| E − F — the SHA-256 entropy term | **+0.00 ± 0.00** | **+0.00 ± 0.00** | 0 of 5 |
| E − D — the reservoir against the control | +1.06 ± 0.77 | +0.99 ± 1.02 | 1 of 5 |
| J − D — the reservoir as a correction | +0.36 ± 0.36 | +0.38 ± 0.18 | 1 of 5, 0 of 5 |
| I − D — explicit 10 s memory | **−0.35 ± 0.16** | **−0.52 ± 0.24** | **5 of 5, 5 of 5** |
| H − E — taking the current window away | +1.28 ± 0.67 | +1.12 ± 0.28 | 0 of 5 |
| D − B — adding the clock | −3.28 ± 2.55 | −2.67 ± 1.56 | 5 of 5 |
| B − A — adding the EEG | −8.20 ± 1.34 | −7.48 ± 0.69 | 5 of 5 |

### The hypothesis was wrong, and this is what the numbers say

**The holographic reservoir did not help.** It is 1.0 s *worse* than the stateless control
and loses in four folds out of five in both regimes. Used only as a correction on top of
the control it still loses by 0.4 s. The sweep behind this is blunter still: the shorter
its memory, the better it scores — at a leak of 0.1 it reaches 80–112 s, at 0.7 it ties.

**The holographic part specifically does not exist as far as measurement goes.** E − F is
`+0.00 ± 0.00` in both regimes: the SHA-256 entropy added to the recurrent matrix's
diagonal gives the same number to the last decimal as leaving it out. What is in there is
an echo-state network.

**Memory does help, but only a little and only if it is short.** One exponential moving
average with a 10 s half-life beats the control by 0.5 s and wins in all ten folds of both
regimes — small, but not noise. Longer memory makes things worse.

**What actually moves the metric** is the combination of classical spectral features
(−7.5 s against the constant), the clock (−2.7 s more) and, above all, fitting the readout
to the competition's weighted absolute error instead of a squared error (−5.8 s on the
validation split). The loss, not the architecture.

![Why a stateless model cannot track the target](figures/perfil_temporal.png)

### Against the published table

The submitted model, scored through the competition's own harness on the public test
split (`code/verify_official.py`): **W-bMAE 55.816** over 40 recordings, against 60.1 for
CTNet, 60.6 for EEGNet and 60.7 for the constant. Inference takes 11.9 s for 5,869
windows; the limit is 80 minutes on one A100. The submission is 7.6 kB and imports only
numpy.

One number that goes against the conclusion, kept because leaving it out would be picking
the data: on that same public test split the reservoir scores **55.741**, 0.08 s *better*
than the submitted model, the opposite of the cross-validation ordering. With 40
recordings that difference means nothing — but it is there.

### Status

Nothing has been submitted. The competition's registration closed on 1 November 2026 and
is the account holder's action, not something this repository can do. The numbers above
are local validation and a local run of the official harness on public data; no
leaderboard position is claimed.


## Reproducing

The public data (NEMAR `nm000287`, 540 recordings, ~1.1 GB) is **not** redistributed here:
it is downloaded from NEMAR under its own terms, and no signal, feature cache or
participant-level artefact from it is committed to this repository.

```bash
pip install -r requirements.txt
python code/prepare_data.py        # downloads NEMAR nm000287 (~1.1 GB)
python code/extract_cache.py       # per-window spectral features -> cache/
python code/run_cv.py --folds 5 --seeds 0 1 2
python code/build_submission.py --out submissions/holografico
python code/verify_official.py --submission submissions/holografico
```

`verify_official.py` is the one that counts: it loads the built `submission.py` the way the
platform does and scores it with the kit's own `predict_streams` and `binned_mae` over the
public test split, rather than with any code of mine.

### Environment notes

- `code/kit_compat.py` works around two version skews between the starting kit (branch
  `main`) and the `neuralbench==0.4.0` the kit itself pins. Both dropped keys only control
  cache permissions and extraction parallelism; neither changes any data or split.
- Window extraction needs `NEURALBENCH_EXTRACT_CPUS=1` on a machine with a modest
  pagefile: the default spawns one torch-loading worker per core.

## Limits, stated plainly

- The clock signal assumes the sealed cohort's streams also start at the beginning of each
  recording, as the public ones do and as the task definition describes. If the sealed
  windows were trimmed differently, the elapsed-time counter would be biased and model E
  would lose part of its advantage; model G is the variant that does not depend on it.
- Everything is validated on the public Muse split only. The sealed cohort is private and
  its scores were not available when this was written.
- The feature front end is deliberately classical (log band powers, ratios, Hjorth
  descriptors). No claim is made that it is the best possible encoder — the point of the
  ablation is the state, not the features.
- Local validation; no leaderboard placement is claimed here beyond what the status section
  states.
