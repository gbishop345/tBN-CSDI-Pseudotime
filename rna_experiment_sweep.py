"""
Shared missing-ratio sweep + multi-trial loop for RNA CSDI (exe_rna.py, exe_rna_reorder.py, exe_mesc.py).

Optional `get_dataloader_fn` on `run_sweep` defaults to `dataset_rna.get_dataloader`; pass another
callable with the same signature to use a different CSV layout default (e.g. `dataset_mesc.get_dataloader`).
"""
import datetime
import json
import os

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

import numpy as np
import torch
import yaml

from main_model import CSDI_RNA
from dataset_rna import get_dataloader
from utils import train, evaluate

DEFAULT_MISSING_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
METRIC_KEYS = ["mse", "rmse", "mae", "crps", "crps_sum"]


def _mr_key(mr: float) -> str:
    return f"{mr:g}"


def trial_seed(base_seed: int, trial_idx: int) -> int:
    """
    Seed for dataset cache + train/valid/test shuffle.

    Must NOT depend on missing-ratio index: if it did, each missing rate would
    use a different 60/20/20 cell split, so curves over missing_ratio would mix
    (easier/harder) test sets with the effect of masking.
    """
    return base_seed + trial_idx


def add_sweep_arguments(parser, *, config_default="base.yaml"):
    parser.add_argument(
        "--config", type=str, default=config_default, help="Path to config file"
    )
    parser.add_argument("--device", default="cuda:0", help="Device for training (e.g., cpu or cuda:0)")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument(
        "--missingratios",
        type=float,
        nargs="+",
        default=DEFAULT_MISSING_RATIOS,
        help=f"Missing ratios to sweep (default: {DEFAULT_MISSING_RATIOS})",
    )
    parser.add_argument(
        "--testmissingratio",
        type=float,
        default=None,
        help="If set, run only this single missing ratio (overrides --missingratios)",
    )
    parser.add_argument("--modelfolder", type=str, default="", help="Folder under ./save/ to load model.pth from")
    parser.add_argument("--nsample", type=int, default=100, help="Number of samples for evaluation")
    parser.add_argument(
        "--ntrials",
        type=int,
        default=5,
        help="Trials per missing ratio (default: 5)",
    )
    parser.add_argument(
        "--white-noise",
        action="store_true",
        help="Vanilla CSDI: i.i.d. Gaussian only (sets use_blue_noise=false; no gen_bn / Cholesky file needed)",
    )
    parser.add_argument(
        "--noise-blend-schedule",
        type=str,
        default=None,
        metavar="SCHED",
        help="Override model.noise_blend_schedule: sigmoid | linear | cumulative | step (default from YAML)",
    )
    parser.add_argument(
        "--noise-blend-step-t",
        type=int,
        default=None,
        metavar="TB",
        help="For schedule step: first TB diffusion indices use all blue (default: num_steps//2 from YAML or code)",
    )
    parser.add_argument(
        "--gamma-start",
        type=float,
        default=None,
        dest="blend_gamma_start",
        metavar="G0",
        help="Override model.gamma_start: Gaussian weight at t=0 is σ(γ_s) for sigmoid schedule (e.g. -6 ≈ nearly all blue)",
    )
    parser.add_argument(
        "--nearly-all-blue-start",
        action="store_true",
        help="Set gamma_start=-6 (~0.25%% Gaussian at t=0 for sigmoid schedule). Ignored if --gamma-start is set.",
    )
    parser.add_argument(
        "--blend-start",
        type=float,
        default=None,
        metavar="W",
        help="For linear/step only: starting Gaussian weight in [0,1] (0=blue … 1=white). Default 0 from config.",
    )
    parser.add_argument(
        "--noise-blend-reverse",
        action="store_true",
        help="Set model.noise_blend_reverse=true: blend uses effective time (T-1)-t for the active schedule",
    )


def run_one_trial(
    trial_idx,
    mr_index,
    missing_ratio,
    config,
    args,
    num_genes,
    num_timepoints,
    file_path: str,
    trial_folder: str,
    get_dataloader_fn=None,
):
    import time as _time

    seed = trial_seed(args.seed, trial_idx)
    os.makedirs(trial_folder, exist_ok=True)

    gdl = get_dataloader_fn if get_dataloader_fn is not None else get_dataloader

    cfg = dict(config)
    cfg["model"] = dict(config["model"])
    cfg["model"]["test_missing_ratio"] = missing_ratio

    train_loader, valid_loader, test_loader, _, _ = gdl(
        seed=seed,
        batch_size=cfg["train"]["batch_size"],
        missing_ratio=missing_ratio,
        file_path=file_path,
    )

    model = CSDI_RNA(
        cfg, args.device,
        target_dim=num_genes,
        num_timepoints=num_timepoints,
    ).to(args.device)

    if args.modelfolder == "":
        train_start = _time.time()
        train(model, cfg["train"], train_loader, valid_loader=valid_loader, foldername=trial_folder)
        training_time = _time.time() - train_start
    else:
        model_path = os.path.join("./save", args.modelfolder, "model.pth")
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        training_time = 0.0

    test_start = _time.time()
    metrics = evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=trial_folder)
    testing_time = _time.time() - test_start

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if metrics is None:
        raise RuntimeError(
            "evaluate() returned None (unexpected). Check utils.evaluate and test_loader."
        )

    return {
        "training_time_seconds": training_time,
        "testing_time_seconds": testing_time,
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "crps": metrics["crps"],
        "crps_sum": metrics["crps_sum"],
    }


def aggregate_trials(all_metrics):
    agg = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in all_metrics]
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg[k] = {"mean": float(np.mean(vals)), "std": std, "values": vals}
    return agg


def run_sweep(
    args,
    file_path: str,
    save_prefix: str,
    dataset_label: str,
    path_json_key: str,
    intro_message: str,
    extra_run_meta=None,
    get_dataloader_fn=None,
):
    args.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.testmissingratio is not None:
        missing_ratios = [args.testmissingratio]
    else:
        missing_ratios = list(args.missingratios)

    config_path = os.path.join("config", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if getattr(args, "white_noise", False):
        config.setdefault("model", {})
        config["model"]["use_blue_noise"] = False
        print("[INFO] --white-noise: model.use_blue_noise=False (vanilla CSDI, no blue-noise Cholesky)\n")

    nbs = getattr(args, "noise_blend_schedule", None)
    if nbs is not None:
        s = nbs.strip().lower()
        allowed = ("sigmoid", "linear", "cumulative", "step")
        if s not in allowed:
            raise ValueError(
                f"--noise-blend-schedule must be one of {list(allowed)}, got {nbs!r}"
            )
        config.setdefault("model", {})
        config["model"]["noise_blend_schedule"] = s
        print(f"[INFO] --noise-blend-schedule: model.noise_blend_schedule={s!r}\n")

    if getattr(args, "noise_blend_step_t", None) is not None:
        config.setdefault("model", {})
        config["model"]["noise_blend_step_t"] = int(args.noise_blend_step_t)
        print(
            f"[INFO] --noise-blend-step-t: model.noise_blend_step_t={config['model']['noise_blend_step_t']}\n"
        )

    if getattr(args, "blend_gamma_start", None) is not None:
        config.setdefault("model", {})
        config["model"]["gamma_start"] = float(args.blend_gamma_start)
        print(f"[INFO] --gamma-start: model.gamma_start={args.blend_gamma_start}\n")
    elif getattr(args, "nearly_all_blue_start", False):
        config.setdefault("model", {})
        config["model"]["gamma_start"] = -6.0
        print("[INFO] --nearly-all-blue-start: model.gamma_start=-6.0 (σ≈0.0025 at t=0)\n")

    if getattr(args, "blend_start", None) is not None:
        w = float(args.blend_start)
        if not (0.0 <= w <= 1.0):
            raise ValueError("--blend-start must be between 0 and 1")
        config.setdefault("model", {})
        config["model"]["noise_blend_w_start"] = w
        print(f"[INFO] --blend-start: model.noise_blend_w_start={w}\n")

    if getattr(args, "noise_blend_reverse", False):
        config.setdefault("model", {})
        config["model"]["noise_blend_reverse"] = True
        print("[INFO] --noise-blend-reverse: model.noise_blend_reverse=True\n")

    base_foldername = f"./save/{save_prefix}_{args.run_id}/"
    os.makedirs(base_foldername, exist_ok=True)
    run_meta = {
        "missing_ratios": missing_ratios,
        "n_trials_per_ratio": args.ntrials,
        "base_seed": args.seed,
        "config_file": args.config,
        path_json_key: file_path,
    }
    if extra_run_meta:
        run_meta = {**run_meta, **extra_run_meta}
    with open(os.path.join(base_foldername, "config.json"), "w") as f:
        json.dump({**run_meta, "full_config": config}, f, indent=4)

    print(intro_message)
    print(f"CSV: {file_path}")
    probe_mr = missing_ratios[0]
    gdl = get_dataloader_fn if get_dataloader_fn is not None else get_dataloader
    _, _, _, num_genes, num_timepoints = gdl(
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        missing_ratio=probe_mr,
        file_path=file_path,
    )
    print(f"Dataset: {num_genes} genes x {num_timepoints} timepoints")
    print(
        f"Missing ratios: {missing_ratios}  |  {args.ntrials} trial(s) each  "
        f"(seeds: base+ trial index for splits/cache; same split across missing ratios)\n"
    )

    by_missing_ratio = {}

    for mr_index, missing_ratio in enumerate(missing_ratios):
        mr_str = _mr_key(missing_ratio)
        print(f"{'#'*60}")
        print(f"# Missing ratio = {missing_ratio}  ({mr_index + 1}/{len(missing_ratios)})")
        print(f"{'#'*60}")

        trial_metrics = []
        for t in range(args.ntrials):
            s = trial_seed(args.seed, t)
            trial_folder = f"{base_foldername}mr_{mr_str}/trial_{t}/"
            print(f"{'='*60}")
            print(f"  Trial {t + 1}/{args.ntrials}  (seed={s})")
            print(f"{'='*60}")
            m = run_one_trial(
                t, mr_index, missing_ratio, config, args,
                num_genes, num_timepoints, file_path, trial_folder,
                get_dataloader_fn=get_dataloader_fn,
            )
            trial_metrics.append(m)
            print(
                f"  RMSE={m['rmse']:.6f}  MAE={m['mae']:.6f}  CRPS={m['crps']:.6f}\n"
            )

        agg = aggregate_trials(trial_metrics)
        trials_payload = [
            {
                "trial_index": i,
                "seed": trial_seed(args.seed, i),
                **trial_metrics[i],
            }
            for i in range(len(trial_metrics))
        ]

        by_missing_ratio[mr_str] = {
            "missing_ratio": missing_ratio,
            "n_trials": args.ntrials,
            "trials": trials_payload,
            "aggregate": agg,
        }

        print(f"--- Missing ratio {missing_ratio}: mean ± std over {args.ntrials} trials ---")
        for k in METRIC_KEYS:
            print(f"  {k.upper():10s}: {agg[k]['mean']:.6f} ± {agg[k]['std']:.6f}")
        print()

    full_summary = {
        "timestamp": args.run_id,
        "base_seed": args.seed,
        "n_trials_per_ratio": args.ntrials,
        "missing_ratios": missing_ratios,
        "dataset": dataset_label,
        path_json_key: file_path,
        "by_missing_ratio": by_missing_ratio,
    }
    if extra_run_meta:
        full_summary = {**full_summary, **extra_run_meta}

    summary_path = os.path.join(base_foldername, "trials_summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=4)

    # Short file: mean ± std per metric per missing ratio only (no per-trial values)
    by_mr_compact = {}
    for mr in missing_ratios:
        mrk = _mr_key(mr)
        agg = by_missing_ratio[mrk]["aggregate"]
        entry = {"missing_ratio": float(mr)}
        for k in METRIC_KEYS:
            entry[k] = {"mean": agg[k]["mean"], "std": agg[k]["std"]}
        trials_list = by_missing_ratio[mrk]["trials"]
        entry["training_time_seconds"] = {
            "mean": float(np.mean([tr["training_time_seconds"] for tr in trials_list])),
            "std": float(np.std([tr["training_time_seconds"] for tr in trials_list], ddof=1))
            if len(trials_list) > 1
            else 0.0,
        }
        entry["testing_time_seconds"] = {
            "mean": float(np.mean([tr["testing_time_seconds"] for tr in trials_list])),
            "std": float(np.std([tr["testing_time_seconds"] for tr in trials_list], ddof=1))
            if len(trials_list) > 1
            else 0.0,
        }
        by_mr_compact[mrk] = entry

    summary_short = {
        "timestamp": args.run_id,
        "dataset": dataset_label,
        path_json_key: file_path,
        "n_trials_per_ratio": args.ntrials,
        "base_seed": args.seed,
        "description": "mean and std over trials per missing ratio (metrics + train/test time); no per-trial arrays",
        "by_missing_ratio": by_mr_compact,
    }
    if extra_run_meta:
        summary_short = {**summary_short, **extra_run_meta}
    summary_short_path = os.path.join(base_foldername, "summary_mean_std.json")
    with open(summary_short_path, "w") as f:
        json.dump(summary_short, f, indent=4)

    print("=" * 60)
    print("OVERALL SUMMARY (mean ± std over trials, by missing ratio)")
    print("=" * 60)
    for mr in missing_ratios:
        mrk = _mr_key(mr)
        agg = by_missing_ratio[mrk]["aggregate"]
        print(f"\n  missing_ratio = {mr}")
        for k in METRIC_KEYS:
            print(f"    {k.upper():10s}: {agg[k]['mean']:.6f} ± {agg[k]['std']:.6f}")
    print("=" * 60)
    print(f"\nFull results JSON: {os.path.abspath(summary_path)}")
    print(f"Short summary JSON: {os.path.abspath(summary_short_path)}\n")

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    shared_path = os.path.join(results_dir, "all_runs_results.json")
    try:
        with open(shared_path, "r") as f:
            all_runs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_runs = []

    for mr in missing_ratios:
        mrk = _mr_key(mr)
        agg = by_missing_ratio[mrk]["aggregate"]
        all_runs.append(
            {
                "missing_ratio": mr,
                "timestamp": args.run_id,
                "dataset": dataset_label,
                "n_trials": args.ntrials,
                "trials_summary_path": os.path.abspath(summary_path),
                "summary_mean_std_path": os.path.abspath(summary_short_path),
                "training_time_seconds": float(
                    np.mean([tr["training_time_seconds"] for tr in by_missing_ratio[mrk]["trials"]])
                ),
                "testing_time_seconds": float(
                    np.mean([tr["testing_time_seconds"] for tr in by_missing_ratio[mrk]["trials"]])
                ),
                **{k: agg[k]["mean"] for k in METRIC_KEYS},
            }
        )

    with open(shared_path, "w") as f:
        if _HAS_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(all_runs, f, indent=4)
        finally:
            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
