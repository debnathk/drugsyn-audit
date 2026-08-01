"""
Combination-overlap diagnostic for drug-combination synergy benchmarks.

The claim of the accompanying paper is that a protocol's name can imply more
than its partition delivers, and that this is invisible unless you count pairs.
Counting pairs is cheap -- no model, no retraining, no new data -- so there is
no good reason not to report it alongside any cold-start result.

    from overlap import combination_overlap
    combination_overlap(train_pairs, test_pairs)   # -> 0.728

Run this file directly for a worked example.
"""
from __future__ import annotations

from typing import Iterable, Hashable, Tuple


def canonical_pair(a: Hashable, b: Hashable) -> Tuple[Hashable, Hashable]:
    """Order-invariant key for an unordered drug pair.

    Synergy is symmetric, so (A, B) and (B, A) are the same combination. If a
    corpus is not already canonicalised, the overlap will be understated --
    the same pair on either side of the split will look like two pairs.
    """
    return (a, b) if a <= b else (b, a)


def combination_overlap(train_pairs: Iterable[Tuple[Hashable, Hashable]],
                        test_pairs: Iterable[Tuple[Hashable, Hashable]]) -> float:
    """Fraction of test combinations that also appear in training.

    Args:
        train_pairs: (drug_a, drug_b) for every training row.
        test_pairs:  (drug_a, drug_b) for every test row.

    Returns:
        Fraction in [0, 1] of *distinct* test pairs also present in training.
        0.0 means the protocol holds out combinations; values near 1.0 mean it
        does not, whatever else it holds out.

    Note this counts distinct pairs, not rows. A row-weighted version answers a
    different question ("what fraction of test measurements involve a seen
    combination") and is usually higher still.
    """
    train = {canonical_pair(*p) for p in train_pairs}
    test = {canonical_pair(*p) for p in test_pairs}
    if not test:
        raise ValueError("test_pairs is empty; overlap is undefined")
    return len(test & train) / len(test)


def overlap_from_dataframe(df, train_idx, test_idx,
                           drug1_col: str = "drug1", drug2_col: str = "drug2") -> float:
    """Convenience wrapper for a pandas corpus plus row-index splits."""
    tr = df.iloc[list(train_idx)]
    te = df.iloc[list(test_idx)]
    return combination_overlap(zip(tr[drug1_col], tr[drug2_col]),
                               zip(te[drug1_col], te[drug2_col]))


if __name__ == "__main__":
    # A screen assays its pair matrix across its whole cell panel. Holding out
    # a cell line therefore removes a context, not a combination -- which is
    # why the figure below is 1.0 and not 0.0.
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    cells = ["HCT116", "A375", "MCF7"]
    rows = [(p, c) for p in pairs for c in cells]

    train = [p for p, c in rows if c != "MCF7"]
    test = [p for p, c in rows if c == "MCF7"]
    print(f"leave-cell-line-out overlap: {combination_overlap(train, test):.3f}")

    train = [p for p, c in rows if p != ("B", "C")]
    test = [p for p, c in rows if p == ("B", "C")]
    print(f"leave-combination-out overlap: {combination_overlap(train, test):.3f}")
