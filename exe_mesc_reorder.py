"""
CSDI on pseudotime-reordered mESC (cells×genes+h CSVs under data/mESC/).

Build reordered CSVs:
  python reorder_rna.py --dataset mesc --method dpt
  python reorder_rna.py --dataset mesc --method slingshot
  python reorder_rna.py --dataset mesc --method phate

Same sweep as exe_rna_reorder.py; uses `dataset_rna` tensors via `dataset_mesc_reorder.get_dataloader`.
Default config `base_mesc.yaml` (same blue-noise defaults as RNA; run `gen_bn.py --dataset mesc` first).
"""
import argparse
import os

from dataset_mesc_reorder import REORDERED_BY_METHOD, get_dataloader
from rna_experiment_sweep import add_sweep_arguments, run_sweep


SAVE_PREFIX_BY_METHOD = {
    "dpt": "mesc_reorder",
    "slingshot": "mesc_reorder_slingshot",
    "phate": "mesc_reorder_phate",
}
DATASET_LABEL_BY_METHOD = {
    "dpt": "mesc_reorder",
    "slingshot": "mesc_reorder_slingshot",
    "phate": "mesc_reorder_phate",
}


def main():
    parser = argparse.ArgumentParser(
        description="CSDI for reordered mESC (pseudotime-aligned cells×genes+h)"
    )
    parser.add_argument(
        "--pseudotime-method",
        choices=("dpt", "slingshot", "phate"),
        default="dpt",
        help="Which reordered CSV to train on. Run reorder_rna.py --dataset mesc with matching --method first.",
    )
    add_sweep_arguments(parser, config_default="base_mesc.yaml")
    args = parser.parse_args()

    method = args.pseudotime_method
    file_path = REORDERED_BY_METHOD[method]
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Missing {file_path!r}. Generate it with:\n"
            f"  python reorder_rna.py --dataset mesc --method {method}"
        )
    run_sweep(
        args,
        file_path=file_path,
        save_prefix=SAVE_PREFIX_BY_METHOD[method],
        dataset_label=DATASET_LABEL_BY_METHOD[method],
        path_json_key="mesc_reordered_csv",
        intro_message=f"Reordered mESC experiment (pseudotime_method={method})",
        extra_run_meta={"pseudotime_method": method, "dataset_preset": "mesc"},
        get_dataloader_fn=get_dataloader,
    )


if __name__ == "__main__":
    main()
