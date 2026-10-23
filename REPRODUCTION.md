# Reproduction profiles

The paper's numerical hyperparameters are the intended specification. The
`paper` profile restores methods found in the original `emopain-snn` recurrent
LIF experiment at `bd50395d` (8 June 2024), with those numerical settings. It is
an auditable restoration candidate. The exact historical final-run identity
has not been recovered, and regenerated aggregate paper scores are not claimed.

## Run the small fixed-settings check

Install `requirements-reproduction.txt` in a Python environment, and prepare
licensed data as described in [DATA_CONTRACT.md](DATA_CONTRACT.md). The local
verification environment uses Python 3.12.14 on Apple Silicon CPU.

```sh
python -m unittest discover -s tests -v
python train.py --data-dir data --output-dir reproduction-runs/paper-small
```

`python reproduce.py` accepts the same arguments. Both entry points default to
`--profile paper`, all three modalities, and only folds `0 18`: six fresh jobs.
Each retains 30 encoder and 25 classifier epochs. In the recovered local data
order these are C60_D.mat (negative) and P92_D.mat (positive). Fold identities
are determined by pickle ordering. See the run's recorded data contract.

No aggregate accuracy, AUC, F1 or MCC is estimated from two selected recordings.
To select a single modality/fold, add `--modalities emg --folds 0`. The full
46-fold × three-modality evaluation requires an explicit `--all-folds` flag;
it is not a parameter search and is outside the small check.

## Explicit configuration

`utils/paper_pipeline.py:paper_config` is the single paper profile:

- Batch-size parameters sEMG/Energy/Angle: 32/8/16.
- Encoder/classifier AdamW LR: .005/.00075; epochs: 30/25.
- Encoder windows: 3000 frames, stride 750; five spike positions.
- Encoder widths 3000/3000, classifier width 500, dropout .5.
- Binarization threshold .99; initial recurrent decay .99; five recurrent steps.
- Original recurrent-LIF trainable decay (`learn_beta=True`).
- Pain labels, all modalities, and fresh encoding.

There is no later batch-size override or hidden classifier learning rate.
`--num-steps` is an explicit sensitivity override, captured in provenance; a
non-five value is not the paper's numerical specification.

## Restored original methods

The original code performs additional normalization per channel in every
window using `(x - window_min) / window_max`. That exact arithmetic is retained;
it is intentionally not replaced with `(x - min) / (max - min)`.

Its weighted replacement sampler draws `2 * batch_size` samples per epoch and
uses these as one training batch. Thus the actual training batch contains
64/16/32 draws for the three modalities. This distinction is part of the
original method, not a second hidden override. Validation retains the configured
batch size. Sampling is repeated during classifier training rather than freezing
one previously sampled encoded training set.

The encoder loss uses the original unweighted spike-position sum and original
MI arithmetic: `joint * log2((joint / px) * pz)`. The rewritten loss instead used
a position-weighted sum and `joint * log2(joint / (px * pz))`. `EncoderLoss` now
names the original mode `snn`; its default `weighted` mode remains available for
archival code. Tensor epsilon handling makes an all-zero window finite. The
original stacked loss's separate Z1-term omission and divisor are retained and
explicitly are not presented as a corrected information estimator.

The classifier follows original `RecurrentClassifier3`: one complete encoded
window is presented at each recurrent step, with recurrent state reset for each
window. It trains and validates on all complete overlapping windows. The
recording decision is the prediction of the final complete window, as in the
original script. Density counts spikes across all complete overlapping windows.
These are different operations from the cleanup classifier that used only the
first spike-position chunks of a concatenated recording.

The feed-forward stacked encoder's outputs and gradients match the pinned
original implementation. The original SLSTM/Alpha experiments are not used.

## Deliberate checkpoint repair

The original best-state dictionaries could alias live tensors and end up
containing later weights. The paper profile now saves independent model states
selected by the mean validation loss over a complete epoch, then reloads them.
This fixes state aliasing and the original partial-window validation selection.
It is a deliberate behavior correction, not a claim of bitwise end-to-end
identity to a buggy historical training run. Selected epochs and complete loss
histories are retained in each fold's `details.json`.

## Provenance, fresh runs and resume

The paper profile never reuses encoded-spike caches. The archived profile also
forces fresh encoding when invoked through this runner. Changing parameters
cannot silently preserve a previous encoder's inputs.

Each output directory has a `provenance.json` containing all effective profiles,
source and input SHA-256 hashes, Python/package versions, CPU backend, seed,
thread count and selected folds/modalities. Separate data-contract files record
loaded tensor hashes, shapes and labels in order. Each fold records exact
train/validation/test indices and selected checkpoints. Models contain no raw
recordings but should still be handled according to the applicable data rules.

After interruption repeat the exact invocation with `--resume`. Completed folds
are skipped and Torch/NumPy RNG state is restored. An interrupted fold restarts
from the preceding completed-fold RNG state. Changed code, configuration, data
or recorded environment cause a rejection. Outputs without valid provenance
are rejected rather than treated as a cache. Use a new directory for a new
experiment. The seed is 1957, initialized once and advanced over selected jobs;
a small subset has a different RNG trajectory from a complete historical run.

## Archival comparison

`--profile archived` retains the earlier recovered cleaned-repository
full-recording classifier, ten recurrent steps by default, record-level
normalization, recording-count sampling, the rewritten weighted loss and the
legacy classifier checkpoint behavior. It does not mean the final SNN paper
run was located. Keep these results separate from `paper` results. Old
output directories made with earlier runner versions cannot be resumed after
source changes; their data remain available for inspection.

Saved predictions in `results/fixmi/emopain_srnn_weighted_sum`, combined using
the historical random-forest stacking notebook at `b8ad64b8`, reproduce 80.43%
accuracy, .679 hard-label AUC, .526 binary F1 and .437 MCC. This is a replay of
saved predictions, not fresh training replication. Their mean spike densities
do not reproduce the complete reported paper row.

The runner creates `metrics.json` only if all three complete 46-fold result
sets exist. The archival ensemble uses three RF bases and an RF final estimator
with seed 42, fitted and scored on the same meta-level rows. It reports binary
and macro F1 separately and explicitly labels the hard-decision AUC and ensemble
resubstitution. This is historical analysis behavior, not nested held-out
ensemble validation. LOO is on recordings, not grouped participant identities.
A revised subject-grouped evaluation is a separate study, not part of this fix.

## Validation

Pinned definitions in `tests/fixtures/snn_reference.py` come from `emopain-snn`
`bd50395d`. Tests compare normalization, encoder/loss/LIF outputs and gradients,
learnable decay, sampling weights and counts, final-window decisions, density,
independent checkpoints, zero windows, resume rejection and a complete tiny fold.
The earlier archived full-recording classifier and pain-loader tests are retained.

## Completed restoration check (23 October 2026)

The default six-job paper-profile check completed in about 18.3 minutes on the
Apple M2 Pro CPU. Every job retained 30 encoder epochs and 25 classifier epochs.
All three modalities predicted class 0 for both selected recordings: the
negative recording was correct, and the positive recording was missed in each
modality. Densities (C60_D / P92_D) were sEMG .300983 / .299598, Energy .553493 /
.462258, and Angle .582522 / .611288.

All 13 regression tests passed both in the working repository and in the
extracted source bundle. The source patch applies to the original stl revision.
Completed runs retained the effective settings, exact split indices, selected
checkpoints and histories. These checks verify restoration and execution; they
do not establish regeneration of the full published score row. A completed-fold
resume was also checked separately without starting any further training.
