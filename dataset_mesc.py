"""
mESC: RamDA expression matrix (`ExpressionData.csv`).

Rows = genes; columns = `RamDA_mESC_<NN>h_<Well>` samples. Tensor layout matches **dataset_rna**:
`(num_cells, T, G)` with **num_cells = max over timepoints of how many samples exist at that time** —
shorter timepoints are **zero-padded** with mask 0, same as unequal cohort sizes in `rna.csv`.

**T** and **G** are inferred from the file (distinct times in column names; gene rows). When **max_genes**
is set, genes are chosen by **sample variance** across all RamDA sample columns, using only **observed**
entries. In this matrix, **0 denotes missing** (the public CSV has no NaNs; zeros are common). NaNs and
infs are treated as missing as well. Genes with fewer than two observed values get variance ``-inf`` and
sort last. Tie-break: stable order by original row index.

Default file: `./data/mESC/ExpressionData.csv`. Cache: `{csv_stem}_gvarZ{G}_missing{ratio}_seed{seed}.pk`.
"""
import os
import pickle
import re

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dataset_rna import random_missing_and_per_gene_normalize

MAX_GENES = 100


def get_mesc_file():
    return "./data/mESC/ExpressionData.csv"


def extract_time_and_id(col_name):
    match = re.match(r"RamDA_mESC_(\d{2}h)_([A-Z]\d{2})", col_name)
    if match:
        time_str = match.group(1)
        cell_id = match.group(2)
        time_val = 0 if time_str == "00h" else int(time_str.replace("h", ""))
        return time_val, cell_id
    return None, None


def _ramda_sample_columns_ordered(metadata):
    """Unique RamDA sample column names in first-seen order."""
    out = []
    seen = set()
    for _, col in metadata:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _mesc_observed_mask(values: np.ndarray) -> np.ndarray:
    """True where value is observed; mESC uses 0 for missing (NaN/inf also unobserved)."""
    x = np.asarray(values, dtype=np.float64)
    return np.isfinite(x) & (x != 0.0)


def _mesc_values_and_mask(block: np.ndarray) -> tuple:
    """Float block (cells × genes): store 0 at missing; mask True only at observed entries."""
    x = np.asarray(block, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    observed = _mesc_observed_mask(x)
    return np.where(observed, x, 0.0), observed


def _top_genes_by_observed_variance(df, gene_rows, sample_cols, k):
    """
    Gene names with highest sample variance across sample_cols; only **observed** entries
    (nonzero finite for mESC; see `_mesc_observed_mask`).
    ``k`` genes returned (or all genes if k >= len(gene_rows)).
    """
    if k is None or k >= len(gene_rows):
        return gene_rows
    if not sample_cols or not gene_rows:
        return gene_rows[:k]

    sub = df.loc[gene_rows, sample_cols].to_numpy(dtype=np.float64)
    n_genes = sub.shape[0]
    variances = np.empty(n_genes, dtype=np.float64)
    for i in range(n_genes):
        row = sub[i]
        obs = row[_mesc_observed_mask(row)]
        if obs.size < 2:
            variances[i] = -np.inf
        else:
            variances[i] = np.var(obs, ddof=1)

    order = np.argsort(-variances, kind="stable")
    chosen_idx = order[:k]
    return [gene_rows[j] for j in chosen_idx]


def parse_mesc_data(df, missing_ratio=0.1, max_genes=None):
    """
    Build (Cells, Time, Genes) like parse_rna_data: for each experimental time h, stack all sample
    columns at that h (any order), pad along the cell axis to max_h |cohort(h)|.

    If ``max_genes`` is not None and smaller than the number of gene rows, keep the top ``max_genes``
    by sample variance over all RamDA columns (observed entries only; **0 = missing** for mESC).
    """
    metadata = []
    for col in df.columns:
        t, _ = extract_time_and_id(col)
        if t is not None:
            metadata.append((t, col))
    if not metadata:
        raise ValueError("No RamDA_mESC_* columns found in expression matrix")

    timepoints = sorted({t for t, _ in metadata})
    by_h = {h: [c for t, c in metadata if t == h] for h in timepoints}
    counts = [len(by_h[h]) for h in timepoints]
    num_cells = max(counts)
    if num_cells == 0:
        raise ValueError("No samples found per timepoint")

    gene_rows = [str(g) for g in df.index]
    n_g_total = len(gene_rows)
    cap = max_genes if max_genes is not None else n_g_total
    n_g = min(n_g_total, int(cap))
    sample_cols_all = _ramda_sample_columns_ordered(metadata)
    gene_rows_use = _top_genes_by_observed_variance(
        df, gene_rows, sample_cols_all, n_g
    )

    observed_values = np.zeros((num_cells, len(timepoints), n_g), dtype=float)
    observed_masks = np.zeros((num_cells, len(timepoints), n_g), dtype=bool)

    for t_idx, h in enumerate(timepoints):
        cols = by_h[h]
        if not cols:
            continue
        sub = df.loc[gene_rows_use, cols]
        block = sub.to_numpy(dtype=np.float64).T
        n_t = block.shape[0]
        vals, m_obs = _mesc_values_and_mask(block)
        observed_values[:n_t, t_idx, :] = vals
        observed_masks[:n_t, t_idx, :] = m_obs

    ov, om, gt = random_missing_and_per_gene_normalize(
        observed_values, observed_masks, missing_ratio
    )
    return ov, om, gt, timepoints


class MESC_Dataset(Dataset):
    def __init__(self, use_index_list=None, missing_ratio=0.1, seed=0, file_path=None):
        np.random.seed(seed)
        self._file_path = file_path if file_path is not None else get_mesc_file()
        base = self._file_path.replace(".csv", "")
        cache_path = f"{base}_gvarZ{MAX_GENES}_missing{missing_ratio}_seed{seed}.pk"

        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(
                f"mESC data not found: {self._file_path!r} (expected ExpressionData.csv)"
            )
        if not os.path.isfile(cache_path):
            data = pd.read_csv(self._file_path, index_col=0)
            parsed = parse_mesc_data(data, missing_ratio, max_genes=MAX_GENES)
            self.observed_values, self.observed_masks, self.gt_masks, self.timepoints = parsed
            with open(cache_path, "wb") as f:
                pickle.dump(parsed, f)
        else:
            with open(cache_path, "rb") as f:
                (
                    self.observed_values,
                    self.observed_masks,
                    self.gt_masks,
                    self.timepoints,
                ) = pickle.load(f)

        self.num_cells = self.observed_values.shape[0]
        if use_index_list is None:
            self.use_index_list = np.arange(self.num_cells)
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        idx = self.use_index_list[org_index]
        return {
            "observed_data": self.observed_values[idx],
            "observed_mask": self.observed_masks[idx],
            "gt_mask": self.gt_masks[idx],
            "timepoints": np.array(self.timepoints, dtype=np.float32),
        }

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, batch_size=16, missing_ratio=0.1, file_path=None):
    np.random.seed(seed)
    fp = file_path if file_path is not None else get_mesc_file()
    full_dataset = MESC_Dataset(missing_ratio=missing_ratio, seed=seed, file_path=fp)
    num_cells = full_dataset.num_cells

    all_indices = np.arange(num_cells)
    np.random.shuffle(all_indices)

    num_train = int(num_cells * 0.6)
    num_valid = int(num_cells * 0.2)

    train_indices = all_indices[:num_train]
    valid_indices = all_indices[num_train : num_train + num_valid]
    test_indices = all_indices[num_train + num_valid :]

    train_dataset = MESC_Dataset(
        use_index_list=train_indices,
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=fp,
    )
    valid_dataset = MESC_Dataset(
        use_index_list=valid_indices,
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=fp,
    )
    test_dataset = MESC_Dataset(
        use_index_list=test_indices,
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=fp,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_timepoints = full_dataset.observed_values.shape[1]
    num_genes = full_dataset.observed_values.shape[2]
    return train_loader, valid_loader, test_loader, num_genes, num_timepoints
