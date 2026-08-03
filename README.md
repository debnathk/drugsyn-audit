# drugsyn-audit

Code, split definitions and manuscript for:

> **Combination Overlap in Drug Synergy Benchmarks:
> An Audit of Cell-Line Holdout Protocols**
> K. Debnath, P. Ghosh — Pacific Symposium on Biocomputing 2027 (submitted)

Combination screens assay a fixed panel of drug pairs across a fixed panel of
cell lines. That makes them dense in *contexts per combination* and sparse in
*distinct combinations* — so holding out a cell line removes a context, not a
combination. Within a single screen, leave-cell-line-out retains **100%** of test
combinations in training, by construction. Pooled across a harmonised DrugComb
corpus the figure is 72.8%.

## The diagnostic

The paper's practical recommendation is one number that costs nothing to compute.
[`overlap.py`](overlap.py) is a standalone implementation with no dependencies:

```python
from overlap import combination_overlap
combination_overlap(train_pairs, test_pairs)   # -> 0.728
```

```bash
python overlap.py     # worked example: 1.000 vs 0.000
```

Report it alongside any cold-start result. It needs no model, no retraining and
no new data — it is a property of the partition.

## Layout

| Path | Contents |
|---|---|
| `overlap.py` | Standalone overlap diagnostic (start here) |
| `src/splits.py` | The six evaluation protocols, with mass-aware group packing |
| `src/baselines.py` | Null models, including **B1** (per-cell + per-pair mean) |
| `splits/` | Split definitions as row indices, gzipped, with corpus fingerprints |
| `results/` | Per-seed metrics backing every table in the paper |
| `manuscript/` | LaTeX source, figures and compiled PDF |

## Reproducing

Splits are persisted as row indices with a **corpus fingerprint** (sha256 over
sorted `(drug_a, drug_b, cell)`), so a partition replays exactly without
redistributing the corpus. The audit and the null model run on CPU in minutes;
only the neural comparators need a GPU.

The corpus itself is not redistributed here — it derives from
[DrugComb](https://drugcomb.org/) v1.5. Cell features come from DepMap, drug
structures from MolFormer embeddings, and drug–target annotations from ChEMBL.

## Protocols

| Protocol | Holds out combinations? |
|---|---|
| `random` | no — 94.0% overlap |
| `leave_cell_out` | **no — 72.8%** |
| `leave_tissue_out` | **no** |
| `leave_study_out` | mostly — 9.8% |
| `leave_drug_out` | yes — 0.0% |
| `leave_combo_out` | yes — 0.0% |

Where a claim concerns combination discovery, prefer `leave_combo_out`; where it
concerns external validity, prefer `leave_study_out`.

## Note on B1

`B1` predicts a per-cell-line mean plus a per-pair mean deviation, both estimated
on training rows only. It uses no structure, targets, pathways or omics. Its role
is diagnostic: a benchmark on which B1 is competitive is one whose signal is
largely recoverable from *which pair* and *which context*, without chemistry.

Under `leave_cell_out` it significantly outperforms both a symmetric MLP and a
4.5M-parameter pathway-aware model. Reporting it costs seconds.

## License

Manuscript © the authors, CC BY-NC 4.0. Code released for review; see the paper
for the citation.
