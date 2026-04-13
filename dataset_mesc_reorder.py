"""
Reordered mESC: training CSVs are **cells × (genes + h)**, same as `dataset_rna`.

Produce them with `reorder_rna.py --dataset mesc --method dpt|slingshot|phate`.
This module is a thin alias around `dataset_rna.get_dataloader` with path constants for exe_mesc_reorder.
"""
from dataset_rna import get_dataloader as _get_dataloader
from reorder_datasets import MESC_OUTPUT_BY_METHOD

REORDERED_BY_METHOD = MESC_OUTPUT_BY_METHOD


def get_dataloader(seed=1, batch_size=16, missing_ratio=0.1, file_path=None):
    if file_path is None:
        raise TypeError(
            "dataset_mesc_reorder.get_dataloader requires file_path (set by exe_mesc_reorder / sweep)."
        )
    return _get_dataloader(
        seed=seed,
        batch_size=batch_size,
        missing_ratio=missing_ratio,
        file_path=file_path,
    )
