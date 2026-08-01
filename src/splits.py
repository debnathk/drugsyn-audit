"""
Split strategies for drug-combination synergy data.

The repo's existing splitter (src/dataset.py:496) only does random + cold_drug
and is bound to TDC. This module is standalone and operates on any DataFrame
with drug/drug/cell columns.

Why this exists at all: `synergy_master.parquet` has 73,167 unique unordered
drug pairs spread over 635,201 rows -- a mean of 8.68 cells per pair, up to 127.
Under a random row split essentially every test pair also appears in train, so a
random-split correlation measures cell transfer, not combination discovery.
`leave_combo_out` is the split that matches the actual use case ("recommend a
partner for drug X") and should carry the headline number.

The other hazard this module is built around: scripts/create_cold_target_split.py
produced a 297,039-row test set against an 11,935-row train set, because it
split *entities* by count while row mass concentrates on a few high-degree
drugs. `greedy_group_pack` packs by row mass instead, and `make_split` asserts
the achieved fractions are sane before returning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_MODES = (
    "random",
    "leave_drug_out",
    "leave_combo_out",
    "leave_cell_out",
    "leave_tissue_out",
    "leave_study_out",
)

COLD_LEVELS = ("one", "both")

# Tissues with too few rows to form a meaningful held-out fold. Measured on the
# covered corpus: skin 56,765 / lung 28,035 / breast 22,650 / ovary 18,511 /
# large_intestine 15,008 / kidney 13,025 / haematopoietic 4,011 / brain 3,807 /
# prostate 3,585, then a cliff to bone 243, urinary_tract 218, soft_tissue 33,
# liver 15, stomach 15, pancreas 6.
DEFAULT_MIN_TISSUE_ROWS = 3500


@dataclass
class SplitResult:
    train: List[int]
    val: List[int]
    test: List[int]
    mode: str
    seed: int
    metadata: dict = field(default_factory=dict)

    @property
    def sizes(self) -> Tuple[int, int, int]:
        return len(self.train), len(self.val), len(self.test)

    def __repr__(self) -> str:
        n = sum(self.sizes) or 1
        return (
            f"<SplitResult {self.mode} seed={self.seed} "
            f"train={len(self.train)} ({len(self.train)/n:.1%}) "
            f"val={len(self.val)} ({len(self.val)/n:.1%}) "
            f"test={len(self.test)} ({len(self.test)/n:.1%})>"
        )


# ---------------------------------------------------------------------------
# fingerprinting
# ---------------------------------------------------------------------------

def corpus_fingerprint(df: pd.DataFrame, drug_a_col: str = "dA",
                       drug_b_col: str = "dB", cell_col: str = "cell") -> str:
    """sha256 over the sorted (drug_a, drug_b, cell) triples.

    Split files store positional indices. Without a fingerprint those indices
    rot silently the moment the underlying parquet changes -- which is a live
    hazard here: results/synergy_split.json holds 15,390 indices into
    data/synergy/drugcomb_synergy.parquet with nothing pinning the corpus.
    """
    keys = (
        df[drug_a_col].astype(str) + "\x1f"
        + df[drug_b_col].astype(str) + "\x1f"
        + df[cell_col].astype(str)
    )
    h = hashlib.sha256()
    for k in sorted(keys.tolist()):
        h.update(k.encode())
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# mass-aware group packing
# ---------------------------------------------------------------------------

def greedy_group_pack(group_sizes: Dict, frac: Sequence[float],
                      seed: int = 42) -> Dict:
    """Assign groups to bins so that BIN ROW MASS approximates `frac`.

    Sorting by mass descending and always placing the next group in the bin with
    the largest remaining deficit is what prevents the degenerate splits that
    naive count-based partitioning produces on a heavy-tailed degree
    distribution.

    Returns {group_key: bin_index}, bins ordered as (train, val, test).
    """
    if not np.isclose(sum(frac), 1.0):
        raise ValueError(f"frac must sum to 1, got {frac} (sum {sum(frac)})")

    rng = np.random.default_rng(seed)
    total = sum(group_sizes.values())
    targets = np.array(frac, dtype=float) * total
    current = np.zeros(len(frac), dtype=float)

    # Sort by mass desc; break ties with the seeded RNG so equal-mass groups
    # don't inherit dict/lexicographic ordering.
    items = list(group_sizes.items())
    jitter = rng.random(len(items))
    items = [x for _, x in sorted(
        zip(jitter, items), key=lambda p: (-p[1][1], p[0])
    )]

    assignment = {}
    for key, size in items:
        deficit = targets - current
        b = int(np.argmax(deficit))
        assignment[key] = b
        current[b] += size

    return assignment


def _pack_by_group(df: pd.DataFrame, keys: pd.Series, frac, seed) -> np.ndarray:
    """-> per-row bin assignment array."""
    sizes = keys.value_counts().to_dict()
    assign = greedy_group_pack(sizes, frac, seed)
    return keys.map(assign).to_numpy()


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def split_summary(df: pd.DataFrame, split: SplitResult, *,
                  drug_a_col="dA", drug_b_col="dB", cell_col="cell",
                  tissue_col="tissue", synergy_col=None) -> dict:
    """Per-bin composition and cross-bin leakage. Print this at the top of every
    training log -- it is the difference between an honest number and a
    memorized one."""
    def bin_stats(idx: List[int]) -> dict:
        if not idx:
            return {"n_rows": 0}
        sub = df.iloc[idx]
        a = sub[drug_a_col].astype(str)
        b = sub[drug_b_col].astype(str)
        pairs = set(zip(np.minimum(a, b), np.maximum(a, b)))
        out = {
            "n_rows": len(sub),
            "n_drugs": len(set(a) | set(b)),
            "n_pairs": len(pairs),
            "n_cells": sub[cell_col].nunique(),
        }
        if tissue_col in sub.columns:
            out["n_tissues"] = sub[tissue_col].nunique()
        if synergy_col and synergy_col in sub.columns:
            v = pd.to_numeric(sub[synergy_col], errors="coerce")
            out["target_mean"] = round(float(v.mean()), 4)
            out["target_std"] = round(float(v.std()), 4)
        return out

    def entities(idx: List[int]):
        sub = df.iloc[idx]
        a = sub[drug_a_col].astype(str)
        b = sub[drug_b_col].astype(str)
        return (
            set(a) | set(b),
            set(zip(np.minimum(a, b), np.maximum(a, b))),
            set(sub[cell_col].astype(str)),
        )

    tr_d, tr_p, tr_c = entities(split.train)
    te_d, te_p, te_c = entities(split.test)

    return {
        "train": bin_stats(split.train),
        "val": bin_stats(split.val),
        "test": bin_stats(split.test),
        "leakage": {
            "drugs_shared_train_test": len(tr_d & te_d),
            "drugs_test_unseen": len(te_d - tr_d),
            "pairs_shared_train_test": len(tr_p & te_p),
            "pairs_test_unseen": len(te_p - tr_p),
            "cells_shared_train_test": len(tr_c & te_c),
            "cells_test_unseen": len(te_c - tr_c),
            "frac_test_pairs_seen_in_train": (
                round(len(tr_p & te_p) / len(te_p), 4) if te_p else None
            ),
        },
    }


def format_split_summary(summary: dict, mode: str) -> str:
    lines = [f"Split audit ({mode}):"]
    for name in ("train", "val", "test"):
        s = summary[name]
        if not s.get("n_rows"):
            lines.append(f"  {name:5s}: empty")
            continue
        lines.append(
            f"  {name:5s}: {s['n_rows']:>8,} rows | {s.get('n_drugs',0):>5} drugs | "
            f"{s.get('n_pairs',0):>7,} pairs | {s.get('n_cells',0):>4} cells"
            + (f" | target {s['target_mean']:+.2f}+-{s['target_std']:.2f}"
               if "target_mean" in s else "")
        )
    lk = summary["leakage"]
    lines.append(
        f"  leakage: {lk['pairs_shared_train_test']:,} of "
        f"{lk['pairs_shared_train_test'] + lk['pairs_test_unseen']:,} test pairs seen in train "
        f"({lk['frac_test_pairs_seen_in_train']}), "
        f"{lk['drugs_test_unseen']} unseen drugs, {lk['cells_test_unseen']} unseen cells"
    )
    # Only warn where pair overlap is a defect. Under leave_cell_out the pairs
    # are *meant* to recur -- the held-out entity is the cell line -- so warning
    # there would just train people to ignore the warning.
    frac = lk["frac_test_pairs_seen_in_train"]
    if mode in ("random", "leave_drug_out") and frac and frac > 0.5:
        lines.append(
            f"  WARNING: {frac:.1%} of test pairs also appear in train. This number "
            f"measures cell transfer, not combination discovery -- do not quote it "
            f"without the leave_combo_out figure beside it."
        )
    if mode == "leave_cell_out" and lk["cells_shared_train_test"]:
        lines.append(
            f"  WARNING: {lk['cells_shared_train_test']} cells appear in both train "
            f"and test, which leave_cell_out is supposed to prevent."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the splitter
# ---------------------------------------------------------------------------

def make_split(
    df: pd.DataFrame,
    mode: str,
    *,
    indices: Optional[Sequence[int]] = None,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    cold_level: str = "one",
    test_tissue: Optional[Iterable[str]] = None,
    val_tissue: Optional[Iterable[str]] = None,
    train_studies: Optional[Iterable[str]] = None,
    val_studies: Optional[Iterable[str]] = None,
    test_studies: Optional[Iterable[str]] = None,
    drug_a_col: str = "dA",
    drug_b_col: str = "dB",
    cell_col: str = "cell",
    tissue_col: str = "tissue",
    study_col: str = "studies",
    synergy_col: Optional[str] = None,
    min_test_frac: float = 0.05,
    max_test_frac: float = 0.25,
    min_tissue_rows: int = DEFAULT_MIN_TISSUE_ROWS,
    check_fractions: bool = True,
) -> SplitResult:
    """Build a train/val/test split.

    `indices` restricts to a subset of positional row indices (typically the
    rows that survived feature-coverage filtering). Returned indices are
    positional into `df`.
    """
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}, got '{mode}'")
    if cold_level not in COLD_LEVELS:
        raise ValueError(f"cold_level must be one of {COLD_LEVELS}, got '{cold_level}'")

    idx = np.asarray(list(indices) if indices is not None else range(len(df)), dtype=int)
    sub = df.iloc[idx]
    rng = np.random.default_rng(seed)
    meta: dict = {"n_input_rows": int(len(idx))}
    discarded = 0

    if mode == "random":
        perm = rng.permutation(len(idx))
        n_tr = int(len(idx) * frac[0])
        n_va = int(len(idx) * frac[1])
        bins = np.empty(len(idx), dtype=int)
        bins[perm[:n_tr]] = 0
        bins[perm[n_tr:n_tr + n_va]] = 1
        bins[perm[n_tr + n_va:]] = 2

    elif mode == "leave_combo_out":
        a = sub[drug_a_col].astype(str)
        b = sub[drug_b_col].astype(str)
        keys = pd.Series(list(zip(np.minimum(a, b), np.maximum(a, b))), index=sub.index)
        bins = _pack_by_group(sub, keys, frac, seed)
        meta["n_groups"] = int(keys.nunique())

    elif mode == "leave_drug_out":
        a = sub[drug_a_col].astype(str)
        b = sub[drug_b_col].astype(str)
        drugs = sorted(set(a) | set(b))
        # Pack drugs by their total row participation.
        mass = pd.concat([a, b]).value_counts().to_dict()
        assign = greedy_group_pack({d: mass.get(d, 0) for d in drugs}, frac, seed)
        ba = a.map(assign).to_numpy()
        bb = b.map(assign).to_numpy()
        # A row belongs to the strictest bin either of its drugs touches.
        bins = np.maximum(ba, bb)
        if cold_level == "both":
            # Keep only rows where BOTH drugs are held out; the mixed rows are
            # discarded rather than pushed into train, which would leak.
            mixed = (bins > 0) & (ba != bb)
            discarded += int(mixed.sum())
            bins = np.where(mixed, -1, bins)
        meta["n_groups"] = len(drugs)
        meta["cold_level"] = cold_level

    elif mode == "leave_cell_out":
        keys = sub[cell_col].astype(str)
        bins = _pack_by_group(sub, keys, frac, seed)
        meta["n_groups"] = int(keys.nunique())

    elif mode == "leave_tissue_out":
        if tissue_col not in sub.columns:
            raise ValueError(f"'{tissue_col}' column required for leave_tissue_out")
        if not test_tissue:
            raise ValueError(
                "leave_tissue_out needs an explicit --test-tissue. The tissue "
                "distribution is far too skewed for a mass-based 80/10/10 "
                f"(eligible: {sorted(eligible_tissues(sub, tissue_col, min_tissue_rows))})"
            )
        counts = sub[tissue_col].astype(str).value_counts()
        eligible = eligible_tissues(sub, tissue_col, min_tissue_rows)
        test_tissue = [str(t) for t in test_tissue]
        val_tissue = [str(t) for t in (val_tissue or [])]
        for t in test_tissue + val_tissue:
            if t not in counts.index:
                raise ValueError(f"tissue '{t}' not present; have {sorted(counts.index)}")
            if t not in eligible:
                raise ValueError(
                    f"tissue '{t}' has only {int(counts[t])} rows (< {min_tissue_rows}); "
                    f"a fold that small measures noise. Eligible: {sorted(eligible)}"
                )
        tis = sub[tissue_col].astype(str)
        bins = np.zeros(len(sub), dtype=int)
        bins[tis.isin(val_tissue).to_numpy()] = 1
        bins[tis.isin(test_tissue).to_numpy()] = 2
        meta.update({"test_tissue": test_tissue, "val_tissue": val_tissue,
                     "eligible_tissues": sorted(eligible)})
        check_fractions = False  # tissue mass is what it is

    elif mode == "leave_study_out":
        if study_col not in sub.columns:
            raise ValueError(f"'{study_col}' column required for leave_study_out")
        st = sub[study_col].astype(str)
        # Rows attributed to several studies cannot belong to one bin; drop them.
        multi = st.str.contains(",").to_numpy()
        discarded += int(multi.sum())
        train_studies = set(train_studies or ["ALMANAC"])
        val_studies = set(val_studies or ["FRIEDMAN"])
        test_studies = set(test_studies or ["ONEIL", "ASTRAZENECA"])
        bins = np.full(len(sub), -1, dtype=int)
        arr = st.to_numpy()
        for i in range(len(sub)):
            if multi[i]:
                continue
            s = arr[i]
            if s in train_studies:
                bins[i] = 0
            elif s in val_studies:
                bins[i] = 1
            elif s in test_studies:
                bins[i] = 2
        discarded += int((bins == -1).sum()) - int(multi.sum())
        meta.update({"train_studies": sorted(train_studies),
                     "val_studies": sorted(val_studies),
                     "test_studies": sorted(test_studies),
                     "n_multi_study_rows_dropped": int(multi.sum())})
        check_fractions = False

    else:  # pragma: no cover - guarded above
        raise AssertionError(mode)

    train = idx[bins == 0].tolist()
    val = idx[bins == 1].tolist()
    test = idx[bins == 2].tolist()

    total = len(train) + len(val) + len(test)
    if total == 0:
        raise ValueError(f"split '{mode}' produced no rows at all")

    achieved = (len(train) / total, len(val) / total, len(test) / total)
    meta.update({
        "frac_requested": list(frac),
        "frac_achieved": [round(x, 4) for x in achieved],
        "n_discarded": discarded,
    })

    if check_fractions and not (min_test_frac <= achieved[2] <= max_test_frac):
        raise AssertionError(
            f"split '{mode}' is degenerate: achieved fractions "
            f"train={achieved[0]:.3f} val={achieved[1]:.3f} test={achieved[2]:.3f}, "
            f"but test must lie in [{min_test_frac}, {max_test_frac}]. "
            f"This is the failure mode that produced the 297,039-vs-11,935 split in "
            f"scripts/create_cold_target_split.py."
        )
    if not train or not test:
        raise AssertionError(
            f"split '{mode}' left an empty bin: train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

    result = SplitResult(train=train, val=val, test=test, mode=mode, seed=seed,
                         metadata=meta)
    result.metadata["summary"] = split_summary(
        df, result, drug_a_col=drug_a_col, drug_b_col=drug_b_col,
        cell_col=cell_col, tissue_col=tissue_col, synergy_col=synergy_col,
    )
    logger.info("%s", result)
    return result


def eligible_tissues(df: pd.DataFrame, tissue_col: str = "tissue",
                     min_rows: int = DEFAULT_MIN_TISSUE_ROWS) -> List[str]:
    counts = df[tissue_col].astype(str).value_counts()
    return sorted(counts[counts >= min_rows].index.tolist())


def leave_one_tissue_out_folds(df: pd.DataFrame, *, tissue_col: str = "tissue",
                               min_rows: int = DEFAULT_MIN_TISSUE_ROWS,
                               **kwargs) -> Iterable[Tuple[str, SplitResult]]:
    """Yield (tissue, split) for each eligible held-out tissue."""
    tissues = eligible_tissues(df, tissue_col, min_rows)
    for i, t in enumerate(tissues):
        val_t = tissues[(i + 1) % len(tissues)]
        yield t, make_split(df, "leave_tissue_out", test_tissue=[t],
                            val_tissue=[val_t], tissue_col=tissue_col,
                            min_tissue_rows=min_rows, **kwargs)


# ---------------------------------------------------------------------------
# persistence -- reuses synergy_eval's save/load verbatim
# ---------------------------------------------------------------------------

def split_path(results_dir, ontology: str, target: str, mode: str,
               seed: int, cold_level: Optional[str] = None):
    from pathlib import Path

    stem = f"pathsyn_{ontology}_{target}_{mode}"
    if mode == "leave_drug_out" and cold_level:
        stem += f"_{cold_level}"
    return Path(results_dir) / "splits" / f"{stem}_{seed}.json"


def save_split(split: SplitResult, path, *, extra_metadata: Optional[dict] = None):
    from src.evaluation.synergy_eval import save_split_indices

    meta = dict(split.metadata)
    meta["mode"] = split.mode
    if extra_metadata:
        meta.update(extra_metadata)
    save_split_indices(split.train, split.val, split.test, str(path),
                       random_seed=split.seed, metadata=meta)


def load_split(path) -> SplitResult:
    from src.evaluation.synergy_eval import load_split_indices

    train, val, test, meta = load_split_indices(str(path))
    return SplitResult(train=train, val=val, test=test,
                       mode=meta.get("mode", "unknown"),
                       seed=meta.get("random_seed", -1), metadata=meta)


def assert_corpus_matches(split: SplitResult, df: pd.DataFrame, **cols):
    """Guard against silently reusing indices against a changed corpus."""
    stored = split.metadata.get("corpus_fingerprint")
    if not stored:
        logger.warning(
            "split has no corpus_fingerprint; cannot verify it matches this "
            "DataFrame. Indices may be stale."
        )
        return
    actual = corpus_fingerprint(df, **cols)
    if stored != actual:
        raise AssertionError(
            f"split was built against a different corpus\n"
            f"  stored: {stored}\n  actual: {actual}\n"
            f"Row indices are positional, so reusing this split would train and "
            f"evaluate on the wrong rows. Rebuild the split."
        )
