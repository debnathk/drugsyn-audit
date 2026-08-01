#!/usr/bin/env python3
"""
Baselines for PathSyn, all evaluated on an identical split file.

The point of B1 in particular: a per-cell + per-pair mean predictor has no
chemistry, no biology and no pathway information at all. Under a random row
split it does well, because ~94% of test pairs also appear in train. Under
leave_combo_out it should collapse. If PathSyn cannot beat B1 by a wide margin
on leave_combo_out, PathSyn is memorizing pairs rather than learning
combination biology -- and no amount of pathway machinery fixes that.

B3 is the number that justifies the whole project: a symmetric MLP over drug
embeddings and cell features, with no ontology, no graph and no target
projection. Everything PathSyn adds has to earn its place against it.

Usage:
    python scripts/run_pathsyn_baselines.py --split-file results/splits/....json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.pathsyn_eval import (
    TARGET_COLUMNS,
    compute_all_metrics,
    fit_target_scaler,
    load_drug_embeddings,
)
from src.evaluation.splits import assert_corpus_matches, load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def b0_global_mean(y_tr, y_te, **_):
    return np.full(len(y_te), float(np.mean(y_tr)))


def b1_cell_pair_mean(df, tr, te, col, **_):
    """Per-cell mean + per-pair deviation. The leakage detector."""
    d = df.iloc[tr]
    g = float(d[col].mean())
    cell_mean = d.groupby(d["cell"].astype(str))[col].mean()

    a = d["dA"].astype(str); b = d["dB"].astype(str)
    # a tuple-valued Series, not a bare list -- pandas reads a list of tuples as
    # a set of column names and raises KeyError on the first pair
    key = pd.Series(list(zip(np.minimum(a, b), np.maximum(a, b))), index=d.index)
    resid = d[col] - d["cell"].astype(str).map(cell_mean).fillna(g)
    pair_dev = resid.groupby(key).mean()

    t = df.iloc[te]
    ta = t["dA"].astype(str); tb = t["dB"].astype(str)
    tkey = pd.Series(list(zip(np.minimum(ta, tb), np.maximum(ta, tb))), index=t.index)
    base = t["cell"].astype(str).map(cell_mean).fillna(g).to_numpy()
    dev = tkey.map(pair_dev).fillna(0.0).to_numpy()
    return base + dev


def _drug_matrix(df, idx, smi2vec, name2smi, dim):
    a = df.iloc[idx]["dA"].astype(str).str.lower()
    b = df.iloc[idx]["dB"].astype(str).str.lower()
    A = np.stack([smi2vec[name2smi[x]] for x in a]).astype(np.float32)
    B = np.stack([smi2vec[name2smi[x]] for x in b]).astype(np.float32)
    return A, B


def _cell_matrix(df, idx, cellf):
    c = df.iloc[idx]["cell"].astype(str).str.lower()
    return np.stack([cellf[x] for x in c]).astype(np.float32)


def b2_gbm(df, tr, te, col, smi2vec, name2smi, cellf, seed=42, **_):
    """Gradient boosting on PCA'd embeddings -- the non-DL reference."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor

    dim = len(next(iter(smi2vec.values())))
    Atr, Btr = _drug_matrix(df, tr, smi2vec, name2smi, dim)
    Ate, Bte = _drug_matrix(df, te, smi2vec, name2smi, dim)
    Ctr, Cte = _cell_matrix(df, tr, cellf), _cell_matrix(df, te, cellf)

    p = PCA(n_components=64, random_state=seed).fit(np.vstack([Atr, Btr]))
    # symmetric encoding so drug order cannot matter
    ftr = np.hstack([p.transform(Atr) + p.transform(Btr),
                     np.abs(p.transform(Atr) - p.transform(Btr)), Ctr])
    fte = np.hstack([p.transform(Ate) + p.transform(Bte),
                     np.abs(p.transform(Ate) - p.transform(Bte)), Cte])

    m = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    m.fit(ftr, df.iloc[tr][col].to_numpy())
    return m.predict(fte)


def b3_symmetric_mlp(df, tr, te, col, smi2vec, name2smi, cellf, seed=42,
                     epochs=30, batch=512, lr=1e-3, **_):
    """Symmetric MLP: no ontology, no graph, no targets. The number to beat."""
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = len(next(iter(smi2vec.values())))

    Atr, Btr = _drug_matrix(df, tr, smi2vec, name2smi, dim)
    Ate, Bte = _drug_matrix(df, te, smi2vec, name2smi, dim)
    Ctr, Cte = _cell_matrix(df, tr, cellf), _cell_matrix(df, te, cellf)

    ytr = df.iloc[tr][col].to_numpy(np.float32)
    sc = fit_target_scaler(ytr)
    ys = ((np.clip(ytr, sc["winsor_lo"], sc["winsor_hi"]) - sc["mean"]) / sc["std"])

    def feats(A, B, C):
        return torch.from_numpy(np.hstack([A + B, np.abs(A - B), C]).astype(np.float32))

    Xtr, Xte = feats(Atr, Btr, Ctr).to(dev), feats(Ate, Bte, Cte).to(dev)
    Ytr = torch.from_numpy(ys.astype(np.float32)).unsqueeze(-1).to(dev)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xtr[j]), Ytr[j]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        p = model(Xte).cpu().numpy().ravel()
    return p * sc["std"] + sc["mean"]


BASELINES = {"B0": b0_global_mean, "B1": b1_cell_pair_mean,
             "B2": b2_gbm, "B3": b3_symmetric_mlp}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--synergy-data", default=str(ROOT / "data/curated/synergy_master.parquet"))
    ap.add_argument("--target", choices=list(TARGET_COLUMNS), default="loewe")
    ap.add_argument("--drug-embeddings", default=str(ROOT / "data/curated/molformer_embeddings.npz"))
    ap.add_argument("--name2smiles", default=str(ROOT / "data/curated/name2smiles.parquet"))
    ap.add_argument("--cell-features", default=str(ROOT / "data/curated/cell_features_general.pt"))
    ap.add_argument("--baselines", default="B0,B1,B2,B3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    col = TARGET_COLUMNS[args.target]
    df = pd.read_parquet(args.synergy_data)
    df = df[(df["dA"].astype(str) != df["dB"].astype(str)) & df[col].notna()].reset_index(drop=True)

    split = load_split(args.split_file)
    assert_corpus_matches(split, df)
    tr, te = np.array(split.train), np.array(split.test)
    logger.info("Split '%s': %d train / %d test", split.mode, len(tr), len(te))

    smi2vec, name2smi, _, _ = load_drug_embeddings(args.drug_embeddings, args.name2smiles)
    cf = torch.load(args.cell_features, map_location="cpu", weights_only=False)
    cellf = cf["features"]

    y_te = df.iloc[te][col].to_numpy()
    y_tr = df.iloc[tr][col].to_numpy()

    results = {}
    for name in args.baselines.split(","):
        name = name.strip()
        if name not in BASELINES:
            logger.warning("unknown baseline %s", name)
            continue
        t0 = time.time()
        pred = BASELINES[name](df=df, tr=tr, te=te, col=col, y_tr=y_tr, y_te=y_te,
                               smi2vec=smi2vec, name2smi=name2smi, cellf=cellf,
                               seed=args.seed)
        m = compute_all_metrics(y_te, pred)
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        logger.info("%-3s PCC %+.4f | SCC %+.4f | RMSE %7.3f | top10%% %.3f | AUC %.3f | %.0fs",
                    name, m["pearson"], m["spearman"], m["rmse"],
                    m["top10pct_enrichment"], m["binary_auc"], m["seconds"])

    out = Path(args.out or ROOT / f"results/baselines_{split.mode}_{args.target}_{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"split_file": str(args.split_file), "split_mode": split.mode,
               "target": args.target, "n_train": len(tr), "n_test": len(te),
               "results": results}, open(out, "w"), indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
