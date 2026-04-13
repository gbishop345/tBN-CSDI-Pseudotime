"""
CSDI on original-order RNA (`./data/rna/rna.csv`).
Same sweep as exe_rna_reorder.py: default 5 missing ratios (0.1–0.9) × 5 trials; see rna_experiment_sweep.py.
"""
import argparse

from rna_experiment_sweep import add_sweep_arguments, run_sweep

# Match dataset_rna.get_rna_file() for cache consistency
RNA_DATA_PATH = "./data/rna/rna.csv"


def main():
    parser = argparse.ArgumentParser(
        description="CSDI for RNA (original CSV ordering, not pseudotime-reordered)"
    )
    add_sweep_arguments(parser)
    args = parser.parse_args()
    run_sweep(
        args,
        file_path=RNA_DATA_PATH,
        save_prefix="rna",
        dataset_label="rna",
        path_json_key="rna_csv",
        intro_message="Standard (non-reordered) RNA experiment",
    )


if __name__ == "__main__":
    main()
