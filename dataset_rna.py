import pickle
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def get_rna_file():
    return "./data/rna/rna.csv"

def get_gene_names(df):
    return [col for col in df.columns if col != "h"]


def random_missing_and_per_gene_normalize(
    observed_values: np.ndarray,
    observed_masks: np.ndarray,
    missing_ratio: float,
) -> tuple:
    """
    Build gt_masks (random subset of observed entries hidden for training) and z-score genes
    using only positions where observed_masks is True. observed_masks returned unchanged.
    Used by ``parse_rna_data`` and ``dataset_mesc.parse_mesc_data``.
    """
    observed_values = np.asarray(observed_values, dtype=float)
    observed_masks = np.asarray(observed_masks, dtype=bool)

    om_flat = observed_masks.reshape(-1).copy()
    obs_indices = np.where(om_flat)[0]
    miss_count = int(len(obs_indices) * missing_ratio)
    if miss_count > 0 and len(obs_indices) > 0:
        miss_indices = np.random.choice(obs_indices, miss_count, replace=False)
        om_flat[miss_indices] = False
    gt_masks = om_flat.reshape(observed_masks.shape)

    cells, time, genes = observed_values.shape
    tmp_values = observed_values.reshape(-1, genes)
    tmp_masks = observed_masks.reshape(-1, genes).astype(bool)  # full observation mask for stats
    mean = np.zeros(genes, dtype=float)
    std = np.zeros(genes, dtype=float)

    for g in range(genes):
        c_data = tmp_values[:, g][tmp_masks[:, g]]
        if len(c_data) > 1:
            mean[g] = c_data.mean()
            std[g] = c_data.std()
        else:
            mean[g], std[g] = 0.0, 1.0
        if std[g] < 1e-8:
            std[g] = 1e-8

    for g in range(genes):
        tmp_values[:, g] = (tmp_values[:, g] - mean[g]) / std[g]
    tmp_values = tmp_values * tmp_masks
    observed_values = tmp_values.reshape(observed_values.shape)

    return (
        observed_values.astype("float32"),
        observed_masks.astype("float32"),
        gt_masks.astype("float32"),
    )


def _top_gene_columns_by_variance(df, gene_cols, k: int, zero_as_missing: bool) -> list:
    """
    Top ``k`` gene column names by descending sample variance over all rows (stable tie-break).
    Observed: finite values; if ``zero_as_missing``, treat 0 as missing (mESC-style).
    """
    if k >= len(gene_cols):
        return list(gene_cols)
    variances = np.empty(len(gene_cols), dtype=np.float64)
    for j, col in enumerate(gene_cols):
        v = df[col].to_numpy(dtype=np.float64, copy=False)
        if zero_as_missing:
            obs = np.isfinite(v) & (v != 0.0)
        else:
            obs = np.isfinite(v) & ~np.isnan(v)
        x = v[obs]
        if x.size < 2:
            variances[j] = -np.inf
        else:
            variances[j] = np.var(x, ddof=1)
    order = np.argsort(-variances, kind="stable")
    return [gene_cols[i] for i in order[:k]]


def parse_rna_data(
    df,
    missing_ratio=0.1,
    max_genes=None,
    zero_as_missing=False,
):
    """
    Creates:
      observed_values: (Cells, Time, Genes)
      observed_masks:  (Cells, Time, Genes)  # from CSV's non-NaNs
      gt_masks:        (Cells, Time, Genes)  # a random subset of observed_masks
    Normalizes 'observed_values' so that only observed points are scaled, others remain 0.
    Supports unequal cells per timepoint: pads shorter timepoints with zeros (mask=0).
    Timepoint count and gene count come only from the dataframe (no fixed T or G).

    If ``max_genes`` is set and smaller than the number of gene columns, keep the top genes
    by variance (same idea as ``dataset_mesc``). Use ``zero_as_missing=True`` for mESC reordered CSVs.
    """
    timepoints = sorted(df["h"].unique())
    genes = get_gene_names(df)
    if max_genes is not None and len(genes) > max_genes:
        genes = _top_gene_columns_by_variance(df, genes, max_genes, zero_as_missing)
    counts = [df[df["h"] == h].shape[0] for h in timepoints]
    num_cells = max(counts)

    observed_values = np.zeros((num_cells, len(timepoints), len(genes)), dtype=float)
    observed_masks = np.zeros((num_cells, len(timepoints), len(genes)), dtype=bool)

    for t_idx, h in enumerate(timepoints):
        block = df[df["h"] == h][genes].values  # (n_t, num_genes)
        n_t = block.shape[0]
        vals = np.nan_to_num(block)
        observed_values[:n_t, t_idx, :] = vals
        observed_masks[:n_t, t_idx, :] = ~np.isnan(block)

    ov, om, gt = random_missing_and_per_gene_normalize(
        observed_values, observed_masks, missing_ratio
    )
    return ov, om, gt, timepoints

class RNA_Dataset(Dataset):
    def __init__(
        self,
        eval_length=None,
        use_index_list=None,
        missing_ratio=0.1,
        seed=0,
        file_path=None,
        max_genes=None,
        zero_as_missing=False,
    ):
        np.random.seed(seed)
        self.eval_length = eval_length
        self._file_path = file_path if file_path is not None else get_rna_file()
        # Cache keyed by path so reordered vs original don't clash
        base = self._file_path.replace(".csv", "")
        if max_genes is not None:
            cache_path = (
                f"{base}_gvar{max_genes}_zm{int(zero_as_missing)}_missing{missing_ratio}_seed{seed}.pk"
            )
        else:
            cache_path = f"{base}_missing{missing_ratio}_seed{seed}.pk"

        if not os.path.isfile(cache_path):
            df = pd.read_csv(self._file_path)
            (self.observed_values,
             self.observed_masks,
             self.gt_masks,
             self.timepoints) = parse_rna_data(
                df,
                missing_ratio,
                max_genes=max_genes,
                zero_as_missing=zero_as_missing,
            )
            with open(cache_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, self.timepoints],
                    f
                )
        else:
            with open(cache_path, "rb") as f:
                (self.observed_values,
                 self.observed_masks,
                 self.gt_masks,
                 self.timepoints) = pickle.load(f)

        self.num_cells = self.observed_values.shape[0]
        if use_index_list is None:
            self.use_index_list = np.arange(self.num_cells)
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        data = self.observed_values[org_index]  # (Time, Genes)
        mask = self.observed_masks[org_index]   # (Time, Genes)
        gt   = self.gt_masks[org_index]         # (Time, Genes)
        data_out = {
            "observed_data": data,
            "observed_mask": mask,
            "gt_mask": gt,
            "timepoints": np.array(self.timepoints, dtype=np.float32)
        }
        return data_out

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(
    seed=1,
    batch_size=16,
    missing_ratio=0.1,
    file_path=None,
    max_genes=None,
    zero_as_missing=False,
):
    np.random.seed(seed)  # Ensure reproducibility
    full_dataset = RNA_Dataset(
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=file_path,
        max_genes=max_genes,
        zero_as_missing=zero_as_missing,
    )
    num_cells = full_dataset.num_cells

    # Shuffle the indices randomly
    all_indices = np.arange(num_cells)
    np.random.shuffle(all_indices)

    # Compute dataset splits
    num_train = int(num_cells * 0.6)
    num_valid = int(num_cells * 0.2)

    train_indices = all_indices[:num_train]
    valid_indices = all_indices[num_train:num_train + num_valid]
    test_indices  = all_indices[num_train + num_valid:]

    # Create datasets using the shuffled indices
    train_dataset = RNA_Dataset(
        use_index_list=train_indices,
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=file_path,
        max_genes=max_genes,
        zero_as_missing=zero_as_missing,
    )
    valid_dataset = RNA_Dataset(
        use_index_list=valid_indices,
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=file_path,
        max_genes=max_genes,
        zero_as_missing=zero_as_missing,
    )
    test_dataset = RNA_Dataset(
        use_index_list=test_indices,
        missing_ratio=missing_ratio,
        seed=seed,
        file_path=file_path,
        max_genes=max_genes,
        zero_as_missing=zero_as_missing,
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Enable shuffle for training
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # observed_values: (num_cells, timepoints, genes)
    num_timepoints = full_dataset.observed_values.shape[1]
    num_genes = full_dataset.observed_values.shape[2]
    return train_loader, valid_loader, test_loader, num_genes, num_timepoints