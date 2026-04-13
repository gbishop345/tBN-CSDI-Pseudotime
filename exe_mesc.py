"""
CSDI on mESC RamDA `ExpressionData.csv` (see `dataset_mesc.get_mesc_file()`).

Same sweep as exe_rna.py: default 5 missing ratios (0.1–0.9) × 5 trials; see rna_experiment_sweep.py.
"""
import argparse

from dataset_mesc import get_dataloader, get_mesc_file
from rna_experiment_sweep import add_sweep_arguments, run_sweep


def main():
    parser = argparse.ArgumentParser(
        description="CSDI for mESC ExpressionData (default data/mESC/ExpressionData.csv; config base_mesc.yaml)"
    )
    add_sweep_arguments(parser, config_default="base_mesc.yaml")
    args = parser.parse_args()
    run_sweep(
        args,
        file_path=get_mesc_file(),
        save_prefix="mesc",
        dataset_label="mesc",
        path_json_key="mesc_expression_csv",
        intro_message="mESC experiment (RamDA ExpressionData matrix)",
        get_dataloader_fn=get_dataloader,
    )


if __name__ == "__main__":
    main()
