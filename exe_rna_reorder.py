"""
CSDI on pseudotime-reordered RNA.

Build reordered CSVs (default preset is `rna`):
  python reorder_rna.py --dataset rna --method dpt   # or slingshot / phate
  dpt:        data/rna/rna_reordered_dpt.csv
  slingshot:  data/rna/rna_reordered_slingshot.csv
  phate:      data/rna/rna_reordered_phate.csv

mESC: `reorder_rna.py --dataset mesc` and `exe_mesc_reorder.py`.

Same sweep as exe_rna.py: default 5 missing ratios × 5 trials; see rna_experiment_sweep.py.
"""
import argparse
import os

from rna_experiment_sweep import add_sweep_arguments, run_sweep

REORDERED_BY_METHOD = {
    "dpt": "data/rna/rna_reordered_dpt.csv",
    "slingshot": "data/rna/rna_reordered_slingshot.csv",
    "phate": "data/rna/rna_reordered_phate.csv",
}
SAVE_PREFIX_BY_METHOD = {
    "dpt": "rna_reorder",
    "slingshot": "rna_reorder_slingshot",
    "phate": "rna_reorder_phate",
}
DATASET_LABEL_BY_METHOD = {
    "dpt": "rna_reorder",
    "slingshot": "rna_reorder_slingshot",
    "phate": "rna_reorder_phate",
}


def main():
    parser = argparse.ArgumentParser(description="CSDI for Reordered RNA (pseudotime-aligned)")
    parser.add_argument(
        "--pseudotime-method",
        choices=("dpt", "slingshot", "phate"),
        default="dpt",
        help="Which reordered CSV to train on (default: dpt). Run reorder_rna.py with matching --method first.",
    )
    add_sweep_arguments(parser)
    args = parser.parse_args()

    method = args.pseudotime_method
    file_path = REORDERED_BY_METHOD[method]
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Missing {file_path!r}. Generate it with:\n"
            f"  python reorder_rna.py --dataset rna --method {method}"
        )
    run_sweep(
        args,
        file_path=file_path,
        save_prefix=SAVE_PREFIX_BY_METHOD[method],
        dataset_label=DATASET_LABEL_BY_METHOD[method],
        path_json_key="reordered_csv",
        intro_message=f"Reordered RNA experiment (pseudotime_method={method})",
        extra_run_meta={"pseudotime_method": method},
    )


if __name__ == "__main__":
    main()
