#!/usr/bin/env python3
"""
Reorder RNA by pseudotime for CSDI (same preprocessing as reorder.ipynb).

Methods:
  - dpt: Scanpy diffusion pseudotime with multi-root consensus (default).
  - slingshot: R Bioconductor Slingshot on the same PCA space used for the kNN graph,
    with clusters = experimental timepoint (`h`) and start cluster = earliest timepoint.
    Requires Python package `rpy2` and R package `slingshot` (BiocManager::install("slingshot")).
  - phate: Python PHATE (Moon et al.) 1D embedding on that same PCA matrix as trajectory ordering;
    orientation fixed by positive correlation with experimental `timepoint_h`.
    Requires `pip install phate`.

**Layouts** (see `--dataset` / auto-detect from `--input`):
  - **wide_cells** — rows = cells, columns = gene names + `h` (default `data/rna/rna.csv`).
  - **mesc_expression** — rows = genes, columns = `RamDA_mESC_<NN>h_<Well>` samples (`ExpressionData.csv`).
    In that file **0 encodes missing** (same as `dataset_mesc`); zeros become NaN for graph/HVG, then
    per-gene mean imputation fills NaNs **after** ``min_cells`` filtering so only observed counts matter.

All methods write the same output shape: **cells × (genes + h)** for `dataset_rna.get_dataloader`.

To add another dataset, extend `DATASET_PRESETS` (default input + output paths + `format` key) or pass
`--input` / `--output` explicitly (with a file whose header matches one of the supported layouts).
"""
import argparse
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from reorder_datasets import (
    DATASET_PRESETS,
    DEFAULT_REORDER_DATASET,
    MESC_OUTPUT_BY_METHOD,
    RNA_OUTPUT_BY_METHOD,
)

DEFAULT_DATASET = DEFAULT_REORDER_DATASET
SEED = 42


def parse_ramda_sample_name(name: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse RamDA_mESC_00h_A04 -> (time_h, well_id). Returns (None, None) if not matched."""
    m = re.match(r"RamDA_mESC_(\d{2})h_([A-Z]\d{2})", str(name))
    if not m:
        return None, None
    hh = int(m.group(1))
    return float(hh), m.group(2)


def detect_format(path: str) -> str:
    """Infer `wide_cells` vs `mesc_expression` from CSV header."""
    hdr = pd.read_csv(path, nrows=0)
    cols = [str(c) for c in hdr.columns]
    if "h" in cols:
        return "wide_cells"
    hdr_i = pd.read_csv(path, index_col=0, nrows=0)
    cols_i = [str(c) for c in hdr_i.columns]
    nchk = min(8, len(cols_i))
    if nchk > 0 and all(parse_ramda_sample_name(cols_i[j])[0] is not None for j in range(nchk)):
        return "mesc_expression"
    raise ValueError(
        f"Could not infer layout for {path!r}: need column 'h' (cells×genes+h) "
        f"or RamDA_mESC_* sample columns with gene index (genes×samples)."
    )


def load_matrix_wide_cells(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load CSV: rows = cells, columns = genes + 'h'. Returns (df, gene_cols)."""
    df = pd.read_csv(path, sep=",")
    gene_cols = [c for c in df.columns if c != "h"]
    df.index = [f"Cell_{i}" for i in range(len(df))]
    return df, gene_cols


def load_matrix_mesc_expression(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load genes × samples matrix; return cells × genes + h (index = sample column names).

    ExpressionData uses **0 for missing** (no NaNs in typical files); those become ``NaN`` here so
    downstream steps do not treat missing as true zero expression.
    """
    raw = pd.read_csv(path, index_col=0)
    gene_cols = [str(g) for g in raw.index]
    sample_cols = [c for c in raw.columns if parse_ramda_sample_name(str(c))[0] is not None]
    if not sample_cols:
        raise ValueError(f"No RamDA_mESC_* columns found in {path!r}")
    sub = raw[sample_cols]
    x = sub.to_numpy(dtype=np.float64).T
    h_list = [parse_ramda_sample_name(str(c))[0] for c in sample_cols]
    cell_ids = [str(c) for c in sample_cols]
    out = pd.DataFrame(x, index=cell_ids, columns=gene_cols)
    out[gene_cols] = out[gene_cols].mask(out[gene_cols] == 0.0, np.nan)
    out["h"] = h_list
    return out, gene_cols


def load_matrix(path: str, fmt: str) -> Tuple[pd.DataFrame, List[str]]:
    if fmt == "wide_cells":
        return load_matrix_wide_cells(path)
    if fmt == "mesc_expression":
        return load_matrix_mesc_expression(path)
    raise ValueError(f"Unknown format {fmt!r}; extend load_matrix() or DATASET_PRESETS.")


def build_annadata(
    df: pd.DataFrame,
    gene_cols: List[str],
    *,
    mesc_zero_as_missing: bool = False,
) -> sc.AnnData:
    """Build AnnData from dataframe (cells × genes).

    Default (wide RNA): non-finite values → 0 so HVG/scale/PCA are finite.

    mESC matrix: ``mesc_zero_as_missing=True`` keeps NaN (missing); ``build_graph`` imputes after
    filtering genes by **observed** entry counts.
    """
    X = df[gene_cols].to_numpy(dtype=np.float64, copy=True)
    if mesc_zero_as_missing:
        X[~np.isfinite(X)] = np.nan
        X[X == 0.0] = np.nan
        adata = sc.AnnData(X)
        adata.uns["mesc_zero_as_missing"] = True
    else:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        adata = sc.AnnData(X)
    adata.var_names = gene_cols
    adata.obs_names = df.index.tolist()
    adata.var_names_make_unique()
    adata.obs["timepoint_h"] = df["h"].values
    adata.obs["well"] = adata.obs_names
    return adata


def _impute_column_mean_nanonly(X: np.ndarray) -> np.ndarray:
    """Return a copy with NaN filled by that gene's mean over finite entries (0 if all-NaN column)."""
    X = np.asarray(X, dtype=np.float64, order="C").copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    nan_rows, nan_cols = np.where(~np.isfinite(X))
    if nan_rows.size:
        X[nan_rows, nan_cols] = col_mean[nan_cols]
    return X


def build_graph(adata: sc.AnnData) -> sc.AnnData:
    """Subset to HVGs, scale, PCA, kNN, diffusion map. Returns subset adata (same cells)."""
    mesc = adata.uns.pop("mesc_zero_as_missing", False)
    if mesc:
        X = np.asarray(adata.X, dtype=np.float64)
        n_obs = np.isfinite(X).sum(axis=0)
        keep = n_obs >= 3
        if not np.any(keep):
            raise ValueError(
                "[graph] mESC: no genes with ≥3 observed (finite) entries; check input matrix"
            )
        adata = adata[:, keep].copy()
        adata.X = _impute_column_mean_nanonly(np.asarray(adata.X, dtype=np.float64))
    else:
        sc.pp.filter_genes(adata, min_cells=3)
    mt_mask = (
        adata.var_names.str.lower().str.startswith("mt-")
        | adata.var_names.str.upper().str.startswith(("MT-", "MT_"))
    )
    ribo_mask = adata.var_names.str.match(r"(?i)^RP[LS]\d")
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000, subset=False)
    keep_mask = adata.var["highly_variable"] & ~mt_mask & ~ribo_mask
    adata = adata[:, keep_mask].copy()

    sc.pp.scale(adata, max_value=10)
    # PCA with arpack: n_comps must be strictly less than min(n_obs, n_vars)
    max_pca = min(adata.n_obs - 1, adata.n_vars - 1, 100)
    max_pca = max(1, max_pca)
    sc.tl.pca(adata, n_comps=max_pca, svd_solver="arpack")
    vr = adata.uns["pca"]["variance_ratio"]
    n_pcs = int(np.clip(np.searchsorted(np.cumsum(vr), 0.90) + 1, 20, 60))
    n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])  # cannot exceed actual components
    print(f"[graph] Using n_pcs={n_pcs}")

    n_neighbors = min(30, adata.n_obs - 1)  # need at least 1 neighbor
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric="cosine")
    sc.tl.diffmap(adata)

    ncomp, labels = connected_components(adata.obsp["connectivities"], directed=False)
    adata.obs["graph_comp"] = labels
    deg = np.asarray(adata.obsp["connectivities"].sum(axis=1)).ravel()
    print(f"[graph] components={ncomp} | degree mean±sd={deg.mean():.1f}±{deg.std():.1f}")
    return adata


def consensus_dpt(adata: sc.AnnData, seed: int = SEED) -> None:
    """Run DPT from multiple roots and set adata.obs['pseudotime_consensus']."""
    rng = np.random.default_rng(seed)
    tp = adata.obs["timepoint_h"].values
    earliest_tp = float(np.nanmin(tp))
    early_idx = np.where(tp == earliest_tp)[0]
    n_roots = max(1, int(np.ceil(len(early_idx) / 2)))
    roots = rng.choice(early_idx, size=n_roots, replace=False)
    print(f"[consensus] earliest={earliest_tp}h | candidates={len(early_idx)} | roots={n_roots}")

    sc.tl.diffmap(adata)
    PT = np.full((adata.n_obs, len(roots)), np.nan, dtype=float)
    for j, r in enumerate(roots):
        adata.uns["iroot"] = int(r)
        sc.tl.dpt(adata, n_dcs=10)
        PT[:, j] = adata.obs["dpt_pseudotime"].values

    pt_med = np.nanmedian(PT, axis=1)
    pt_cons = (pt_med - np.nanmin(pt_med)) / (np.nanmax(pt_med) - np.nanmin(pt_med) + 1e-12)
    adata.obs["pseudotime_consensus"] = pt_cons
    adata.obs["pseudotime_consensus_std"] = np.nanstd(PT, axis=1)
    adata.obsm["pseudotime_roots_matrix"] = PT
    print("[consensus] wrote adata.obs['pseudotime_consensus']")


def slingshot_pseudotime(adata: sc.AnnData) -> None:
    """
    Run Slingshot (R) on the same PCA coordinates used for neighbors.

    Cluster labels are stringified `timepoint_h` values (``t_<value>``); starting cluster
    matches the earliest timepoint, analogous to DPT roots sampled from that time.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import StrVector, default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError as e:
        raise ImportError(
            "Slingshot reordering requires rpy2. Install with: pip install rpy2"
        ) from e

    try:
        ro.r("library(slingshot)")
    except Exception as e:
        raise RuntimeError(
            "Could not load R package 'slingshot'. Install in R, e.g.:\n"
            "  install.packages('BiocManager'); BiocManager::install('slingshot')\n"
            f"Underlying error: {e}"
        ) from e

    n_pcs = int(adata.uns["neighbors"]["params"]["n_pcs"])
    X = np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float64)
    tp = np.asarray(adata.obs["timepoint_h"], dtype=float)
    earliest = float(np.nanmin(tp))
    labels = np.array([f"t_{v:g}" for v in tp], dtype=object)
    start_cluster = f"t_{earliest:g}"

    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv["rd"] = ro.conversion.py2rpy(X)
    ro.globalenv["cl"] = StrVector([str(x) for x in labels])
    ro.globalenv["start_clus"] = StrVector([start_cluster])

    ro.r(
        "sds <- slingshot::slingshot(rd, clusterLabels = cl, start.clus = start_clus[[1]])"
    )
    ro.r("pst <- slingshot::slingPseudotime(sds)")
    ro.r("pst_mat <- as.matrix(pst)")

    with localconverter(default_converter + numpy2ri.converter):
        PT = np.asarray(ro.conversion.rpy2py(ro.globalenv["pst_mat"]), dtype=float)

    if PT.ndim == 1:
        PT = PT.reshape(-1, 1)

    pt_med = np.nanmedian(PT, axis=1)
    all_nan_row = ~np.isfinite(pt_med)
    if np.any(all_nan_row):
        pt_med[all_nan_row] = np.nanmean(PT[all_nan_row, :], axis=1)
    if np.any(~np.isfinite(pt_med)):
        raise RuntimeError(
            "Slingshot returned non-finite pseudotime for some cells; check clusters / lineages."
        )

    lo, hi = np.nanmin(pt_med), np.nanmax(pt_med)
    pt_cons = (pt_med - lo) / (hi - lo + 1e-12)
    adata.obs["pseudotime_consensus"] = pt_cons
    adata.obs["pseudotime_consensus_std"] = np.nanstd(PT, axis=1)
    adata.obsm["slingshot_pseudotime_matrix"] = PT
    adata.uns["slingshot_start_cluster"] = start_cluster
    print(
        f"[slingshot] start.clus={start_cluster!r} | lineages={PT.shape[1]} | "
        "wrote adata.obs['pseudotime_consensus']"
    )


def phate_pseudotime(adata: sc.AnnData, seed: int = SEED) -> None:
    """
    1D PHATE embedding on the same PCA coordinates used for Scanpy neighbors (fair vs Slingshot).

    Uses knn equal to the neighbor graph's n_neighbors. Orientation: flip so Spearman with
    `timepoint_h` is non-negative when defined.
    """
    try:
        import phate
    except ImportError as e:
        raise ImportError(
            "PHATE reordering requires the phate package. Install with: pip install phate"
        ) from e

    n_pcs = int(adata.uns["neighbors"]["params"]["n_pcs"])
    knn = int(adata.uns["neighbors"]["params"]["n_neighbors"])
    X = np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float64)
    tp = np.asarray(adata.obs["timepoint_h"], dtype=float)

    op = phate.PHATE(
        n_components=1,
        knn=knn,
        random_state=seed,
        n_jobs=1,
        verbose=0,
    )
    emb = np.asarray(op.fit_transform(X), dtype=float).ravel()

    rho, _ = spearmanr(emb, tp, nan_policy="omit")
    if np.isfinite(rho) and rho < 0:
        emb = -emb

    lo, hi = np.nanmin(emb), np.nanmax(emb)
    pt_cons = (emb - lo) / (hi - lo + 1e-12)
    adata.obs["pseudotime_consensus"] = pt_cons
    adata.obs["pseudotime_consensus_std"] = 0.0
    adata.obsm["X_phate_1d"] = emb.reshape(-1, 1)
    adata.uns["phate_params"] = {"knn": knn, "n_components": 1, "random_state": seed}
    print(f"[phate] knn={knn} | n_components=1 | wrote adata.obs['pseudotime_consensus']")


def save_reordered(
    df: pd.DataFrame,
    gene_cols: List[str],
    adata: sc.AnnData,
    output_path: str,
) -> None:
    """Write cells x (genes + h) CSV ordered by pseudotime."""
    cell_order = adata.obs.sort_values("pseudotime_consensus").index.tolist()
    print(f"Total cells to reorder: {len(cell_order)}")

    original_df = df[gene_cols].T
    original_df.columns = df.index
    reordered_df = original_df[cell_order]
    reordered_cells = reordered_df.T
    reordered_cells["h"] = adata.obs.loc[cell_order, "timepoint_h"].values

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    reordered_cells.to_csv(output_path, index=False)
    print(f"Saved to: {output_path} (cells x [genes + h])")


def resolve_input_output_format(
    dataset: str,
    input_path: Optional[str],
    output_path: Optional[str],
    method: str,
) -> Tuple[str, str, str]:
    """
    Returns (input_path, output_path, format_key).
    `dataset` is a key in DATASET_PRESETS, or 'auto'.
    """
    if input_path is not None:
        fmt = detect_format(input_path)
        inp = input_path
        if output_path is not None:
            return inp, output_path, fmt
        out_map = (
            MESC_OUTPUT_BY_METHOD if fmt == "mesc_expression" else RNA_OUTPUT_BY_METHOD
        )
        return inp, out_map[method], fmt

    if dataset == "auto":
        for name, spec in DATASET_PRESETS.items():
            cand = spec["default_input"]
            if os.path.isfile(cand):
                fmt = spec["format"]
                inp = cand
                out = spec["output_by_method"][method]
                if output_path is not None:
                    out = output_path
                return inp, out, fmt
        raise FileNotFoundError(
            "No preset input file found for --dataset auto; pass --input explicitly. "
            f"Tried: {[s['default_input'] for s in DATASET_PRESETS.values()]}"
        )

    if dataset not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown --dataset {dataset!r}; choose from {list(DATASET_PRESETS)} or 'auto'."
        )
    spec = DATASET_PRESETS[dataset]
    inp = spec["default_input"]
    fmt = spec["format"]
    out = output_path if output_path is not None else spec["output_by_method"][method]
    return inp, out, fmt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorder RNA by pseudotime (DPT, Slingshot, or PHATE); shared HVG/PCA/kNN/diffmap preprocessing."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(list(DATASET_PRESETS.keys()) + ["auto"]),
        default=DEFAULT_DATASET,
        help=(
            "Which preset paths/layout to use when --input is omitted (default: rna). "
            "'mesc' = ExpressionData genes×samples; 'auto' = first existing preset input."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("dpt", "slingshot", "phate"),
        default="dpt",
        help="Pseudotime algorithm after shared preprocessing (default: dpt)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV (overrides --dataset default; layout auto-detected)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output cells×(genes+h) CSV (default: preset path for dataset + method)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for DPT root sampling")
    args = parser.parse_args()

    inp, out, fmt = resolve_input_output_format(
        args.dataset, args.input, args.output, args.method
    )

    sc.settings.verbosity = 2

    print(f"Loading: {inp}  (layout={fmt})")
    df, gene_cols = load_matrix(inp, fmt)
    print(f"Shape: {df.shape} (cells x [genes + h]), genes: {len(gene_cols)}")

    print("Building AnnData and graph (HVGs, PCA, diffusion map)...")
    adata = build_annadata(
        df, gene_cols, mesc_zero_as_missing=(fmt == "mesc_expression")
    )
    adata = build_graph(adata)

    if args.method == "dpt":
        print("Consensus DPT...")
        consensus_dpt(adata, seed=args.seed)
    elif args.method == "slingshot":
        print("Slingshot (R) on shared PCA...")
        slingshot_pseudotime(adata)
    else:
        print("PHATE 1D on shared PCA...")
        phate_pseudotime(adata, seed=args.seed)

    print("Writing reordered CSV...")
    save_reordered(df, gene_cols, adata, out)
    print(f"Done ({args.method}, layout={fmt}). Output: {out}")


if __name__ == "__main__":
    main()
