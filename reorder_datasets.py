"""
Shared presets for `reorder_rna.py`: default input paths, output paths per pseudotime method, and layout keys.

Add a new dataset by appending to `DATASET_PRESETS` and implementing the corresponding loader in `reorder_rna.load_matrix`.
Reordered mESC outputs are listed in `MESC_OUTPUT_BY_METHOD` (used by `dataset_mesc_reorder`).
"""
from typing import Any, Dict

RNA_OUTPUT_BY_METHOD = {
    "dpt": "data/rna/rna_reordered_dpt.csv",
    "slingshot": "data/rna/rna_reordered_slingshot.csv",
    "phate": "data/rna/rna_reordered_phate.csv",
}

MESC_OUTPUT_BY_METHOD = {
    "dpt": "data/mESC/mesc_reordered_dpt.csv",
    "slingshot": "data/mESC/mesc_reordered_slingshot.csv",
    "phate": "data/mESC/mesc_reordered_phate.csv",
}

# name -> { default_input, output_by_method, format }
DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "rna": {
        "default_input": "data/rna/rna.csv",
        "output_by_method": RNA_OUTPUT_BY_METHOD,
        "format": "wide_cells",
    },
    "mesc": {
        "default_input": "data/mESC/ExpressionData.csv",
        "output_by_method": MESC_OUTPUT_BY_METHOD,
        "format": "mesc_expression",
    },
}

DEFAULT_REORDER_DATASET = "rna"
