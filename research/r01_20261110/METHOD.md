# Chronological Explicit Memory — Track 03 sealed entry

Prepared 10 November 2026 by Francisco Angulo de Lafuente (Agnuxo).

## Model

Predict seconds to the first N2 epoch from the current and past EEG windows
only. The inference source is byte-identical to the recovered explicit-memory
model; the weights are newly fitted on complete chronological training streams.

Each raw Muse window has four channels (TP9, AF7, AF8, TP10), 128 Hz and
five seconds: shape (B,4,640), B=1 in the streamed evaluation. The front end
clips amplitudes at +/-500 microvolts, centers each window, and computes
72 spectral/Hjorth/clipping features using 256-sample Hann Welch segments
with 50% overlap. The readout also receives 72 exponential moving averages
with half-life two windows (10 seconds), five elapsed-time features and a
training-only conditional-median latency prior. A standardized linear readout
is initialized by weighted ridge (alpha = 10), then fitted by 12 metric-weighted
IRLS updates. It returns one finite float per window, clipped to [0,600] seconds.

All state resets at each recording. Elapsed time is inferred solely by
counting the supplied windows. No future samples, stream duration, labels,
participant identifiers, cohort membership or private metadata are read at
inference. NumPy and the genuine supplied CompetSolver are the only runtime
imports. Weights are loaded from meta["submission_dir"]; inference performs
no writes there, installation, training, network request or download.

## Public training data, correction and disclosures

Only the original public training split of Interaxon2026Muse / NEMAR
nm000287 is used. The public release has 500 training and 40 public-test
recordings; the task loader stages 412 training and 88 validation recordings.
This model uses the 412 training recordings, 170 participants and 61,303 windows.
The validation and public-test loaders were not iterated for this repair,
study or final fit, and no sealed data or feedback were used to select it.
No external corpus, pretraining, pretrained weights, foundation model or
synthetic training augmentation is used.

An audit found that the historical feature cache had been collected by
iterating the training RegressionBinSampler. It contained replacement draws
in nonchronological order: 61,303 rows but only 30,086 distinct(record,onset)
pairs, with 31,217 duplicates and gaps in 411 of 412 recordings after deduplication.
Fitting the EMA directly in that order did not reproduce the inference
context. Sorting and deduplicating could not recover missing windows.

We therefore extracted EVERY actual training-dataset index exactly once,
without the sampler, and sorted by recording/onset. All 412 streams start
at 0 and proceed in five-second steps without gaps or duplicates. All 61,303
rows of the old sampled cache match their corresponding recovered windows
exactly, including features and targets. The old sources, cache and model
are retained unchanged. New derived features remain private; no raw signals,
participant-level features, labels or original identifiers ship in this ZIP.
The public data retain their CC-BY-NC-SA-4.0 terms; the included Apache-2.0
licence applies to the author's code and does not replace the data terms.

## Prospective comparison and selection

Before outcomes, EEG-R01-v2 fixed five participant-held-out folds (unseen)
and five recording-held-out folds retaining their participants in training
(seen), fold seed 20261110. Every prior, scaler and model is fitted on foldtrain
only. H repeats the historical sampled-cache fitting procedure, L fits the
same IRLS model on complete chronological streams, and N adds a fixed bounded
MLP residual, averaged equally over seeds 11, 29 and 47. All three are evaluated on
the SAME complete chronological held-out recordings: 412 unseen and 91 seen.

The preregistered eligibility criteria versus H are: macro gain at least
0.5 seconds; no point regression in either regime; and positive lower bound
of a paired participant cluster 97.5% bootstrap interval (10,000 replicates,
seed 20261110). Participants share their bootstrap multiplicity across both
regimes. Bonferroni adjusts the two candidate comparisons at a nominal
family level of 5%; bootstrap coverage is approximate. If both pass, choose
the larger macro gain, breaking an exact tie toward the smaller linear model.

| Model | Unseen W-bMAE(s) | Seen W-bMAE(s) | Equal-regime macro(s) | Eligibility |
|---|---:|---:|---:|---|
| H, historical sampled fit | 51.033127 | 50.824883 | 50.929005 | comparator |
| L, chronological IRLS | 49.132254 | 48.700529 | 48.916391 | all gates pass; selected |
| N, nonlinear residual | 51.062118 | 46.425222 | 48.743670 | unseen regression and CI gates fail |

L gains 2.012614 seconds, interval [1.172111, 2.831942] seconds. N's interval is
[-0.344441, 4.485230] seconds and it regresses 0.028991 seconds on unseen people;
its lower point macro score is not enough to qualify. The negative result
is retained; no thresholds, folds, seeds or hyperparameters were changed.
An independent reviewer reproduced all 10,000 participant resamples by direct
record duplication, agreeing with means and interval endpoints to 1e-12.

These are CV results on the public TRAINING release, with historical model
development on that release. They are not an independent new-cohort test,
an official sealed score or a leaderboard claim. The historical 55.816
offline public-test result belongs to the old weights; it is not a score
for this corrected entry. Earlier cached-CV claims are superseded by this
chronological comparison and must not be treated as streaming-parity evidence.

## Compute and reproducibility

Sequential feature reconstruction took 172.928 CPU seconds on Windows using
NeuralBench/neuralset/neuralfetch/neuraltrain 0.4.0, NumPy 2.5.4, PyTorch 2.6.0,
CUDA disabled and four threads. The full prospective comparison took 72.008
seconds on the server including its CPU H/L fits and 30 fixed MLP fits on one
RTX 3090; reserved VRAM peaked at 0.135 GiB and process RAM at 1.843 GiB. The
earlier resource pilot used no held-out predictions or scores.

The SELECTED L model is trained on CPU only. A second deterministic CPU fit
reproduced every saved parameter bit-for-bit in 2.252 seconds. The first fit
completed and saved its weights; its first parity check failed because of an
incorrect local benchmark import path, before loading the model. The correct
path was used for the successful verification; no weights or training data
were changed to fix that environment issue.

Actual loaded Solver predictions on 12 synthetic windows across two recordings
match the offline fitted model within 5.69e-14 seconds (required atol 1e-9, rtol 0).
This is a parity/contract check, not a measure of EEG predictive utility.
Before official upload, this exact ZIP must also complete the actual public
ingestion AND scoring programs using read-only submission files and matched
synthetic inputs. That receipt is stored outside the ZIP under its archive
SHA, avoiding a circular archive hash. Public-worker integration does not
establish equivalence with the sealed Docker image or guarantee sealed runtime.

The local worker uses actual Benchopt 1.10.0, Python 3.13, NumPy 2.5.2,
PyTorch 2.14.1+cu130, pandas 3.1.0, scikit-learn 1.9.1 and pyarrow 26.0.0, with
CUDA disabled for this NumPy model. The current official limit is 60 minutes
on one A100, one submission per day and five total per team/track. Sealed
W-bMAE is computed per recording in bins [0,40), [40,90), [90,300), [300,600]
with weights 10, 5, 3 and 1, averaged within seen/unseen groups and then equally
between groups. The official result is unknown until Codabench finishes.

## Source and audit

Code and original records:
https://github.com/Agnuxo1/holographic-reservoir-sleep-onset
Original inference/training code pin: e7dd3f09d68b6b26067d514e7b528f17ca90d641.
Original and repaired model/input/source hashes: SOURCE_RECORD.json.

Official competition: https://www.codabench.org/competitions/17983/
Current contract pin:
https://github.com/neural-interfaces26/2026-competition/blob/79b44530adda3ffa687dded0046d3424caed9266/codabench/pages/participate.md
Scientific FAQ:
https://github.com/neural-interfaces26/2026-competition/blob/79b44530adda3ffa687dded0046d3424caed9266/codabench/pages/scientific_faq_sleep.md
Data: https://doi.org/10.82901/nemar.nm000287

The organiser's source is referenced, not redistributed. Training code,
configuration, inference, corpus and compute disclosures are preserved for
the required finalist reproducibility audit. The original trained model and
negative studies are retained as research records.

<!-- author-background:start -->
## Author background

**Francisco Angulo de Lafuente**, independent AI researcher · [ORCID](https://orcid.org/0009-0001-1634-7063) · [arXiv](https://arxiv.org/a/angulodelafuente_f_1.html) · [Google Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ) · [ResearchGate](https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3) · [Figshare](https://figshare.com/authors/Francisco_Angulo_de_Lafuente/22601234) · [Wikipedia](https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente)

Earlier research at ECOFA focused on biotechnology for clean energy and on second-generation biofuels obtained by microbiological processing of organic waste. Two Spanish patents document that work:

- **ES 2273594 B1**, *Process for obtaining fuels from organic waste*. Filed 5 September 2005, grant published 16 February 2008. [OEPM](https://www.oepm.es/pdf/ES/0000/000/02/27/35/ES-2273594_B1.pdf) · [Google Patents](https://patents.google.com/patent/ES2273594B1/en)
- **ES 2341194 B1**, *Method for obtaining paraffin by a biological process with microorganisms from organic matter, and its refining for use as fuel*. Filed 11 September 2007, grant published 26 April 2011. Also published as EP2196519 and WO2009034217. [OEPM](https://www.oepm.es/pdf/ES/0000/000/02/34/11/ES-2341194_B1.pdf) · [Google Patents](https://patents.google.com/patent/ES2341194B1/en)

These are historical publication and grant records under the inventor name Francisco Angulo Lafuente, and both have since lapsed. They are cited as evidence of a research trajectory and make no statement about current enforceability.
<!-- author-background:end -->
