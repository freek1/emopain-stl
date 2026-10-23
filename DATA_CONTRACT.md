# Recovered pain-data contract

Use EmoPain only under the applicable access agreement. Raw recordings and
prepared pickle files are not included in the source bundle. The currently
available prepared files were verified against the original `emopain-snn`
`data/read_data.py` extraction; all retained arrays and labels matched.

The pain loader requires `data_Cs.pickle` and `data_Ps.pickle`, each a list of
recording dictionaries. Each dictionary contains `protective_label` (the
historical field name, containing a pain-intensity class), four `sEMG_probe_N`
arrays, twelve `joint_angle_N` arrays and twelve `joint_energy_N` arrays.
Do not substitute `*_protect_kappa.pickle`: those encode protective behaviour.
`load_data_emopain(..., task='pain')` selects pain files explicitly; the public
loader retains an explicit protective-task option for other work.

## Cohort and labels

The recovered raw cohort contains 69 recordings from 43 distinct people:
40 recordings from 25 controls and 29 from 18 patients. Original filtering
removes 23 control recordings with all four sEMG channel sums equal to zero.
The resulting 46 entries are recordings, from 30 distinct people: 17 control
recordings and 29 patient recordings. They are not 46 distinct participants.

Controls receive label 0. For patients, take the NaN-ignoring mean of the 15
self-report fields below, then classify strictly greater than 5 as label 1:

- `N pn bal`, `N pn sit`, `N pn reach`, `N pn stand`, `N pn sitst`, `N pn bend`, `N pn walk`.
- The corresponding seven `D pn ...` fields.
- `pn end`.

Do not add `pn arrive` or `pn sensors`; these change at least one recovered
patient's class. The original extractor also has a rater fallback for missing
self-report rows; preserve that branch if rebuilding its input data.

The prepared local cohort has 34 negative and 12 positive recording labels.
Recordings from the same person can share a class and occur in different folds.
The runner validates this recovered contract rather than silently treating a
43-entry cohort as the same experiment.

## Ordering and signal processing

Controls precede patients. Preserve the prepared pickle order: the original
extractor used filesystem enumeration, not a documented alphabetical order.
Changing order changes folds and RNG consumption. The local prepared files have
SHA-256 values:

- data_Cs.pickle: `9226521cdce3e2ee70100bacec4bb81662f5ea87316e60b9bce92ce67431eb72`
- data_Ps.pickle: `52a3e1b6a1f29fd363fdab1a06932033df8da6ac12c01f963f1b18bc04ea7fd0`.

Hashes identify these exact local pickle bytes, not a universal official dataset
release. Independently serialized equivalent arrays can have different file
hashes. The runner also records hashes of the loaded normalized tensors.

For recordings longer than 18,000 frames, the loader removes 6,000 frames from
each end, then pads recordings to the maximum retained length (17,995 locally).
Each recording/channel is min-max normalized. The paper profile additionally
applies the original window normalization described in REPRODUCTION.md.

Original raw-column extraction uses zero-based sEMG columns 78..81, Angle
134..145 and Energy 137..148. The Angle/Energy ranges overlap by nine columns.
This is retained for historical comparison; replacement indices have not been
verified against an authoritative feature schema and must not be guessed.

Each run records the label vector, tensor shape/hash and exact fold split
indices. If rebuilding preprocessing, retain a source-file-to-recording-index
manifest alongside the pickles and compare arrays and labels before training.
