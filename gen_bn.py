"""Blue-noise masks and Cholesky for CSDI. Needs numpy<2 (see requirements.txt).

Masks and Cholesky factors are written under ``blue_noise/`` with suffix ``_rna`` or ``_mesc``.
Grid shape is **(n_genes, n_timepoints)** to match CSDI_RNA noise layout (tile_k × tile_l after permute).

Usage:
  python gen_bn.py --dataset rna [--input path/to.csv]
  python gen_bn.py --dataset mesc [--input path/to/ExpressionData.csv]
"""
import argparse
import concurrent.futures
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
BLUE_NOISE_DIR = os.path.join(_ROOT, "blue_noise")

DEFAULT_RNA_CSV = os.path.join(_ROOT, "data", "rna", "rna_reordered_dpt.csv")


def _resolve_csv_path(path: str) -> str:
    """Interpret paths relative to repo root when not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_ROOT, path.lstrip("./")))

NUM_REF_MASKS = 10000
INIT_TEMP = 5.0
FINAL_TEMP = 0.1
ANNEALING_RATE = 0.995
NUM_ITERATIONS = 5000
POWER_EXP = 2.0
ADAPTIVE_DECAY = 500
NBINS = 50


def blue_noise_filenames(dataset: str) -> tuple:
    """Return (masks_npz, chol_pt) under ``blue_noise/``; dataset is ``rna`` or ``mesc``."""
    assert dataset in ("rna", "mesc")
    os.makedirs(BLUE_NOISE_DIR, exist_ok=True)
    suf = dataset
    masks = os.path.join(BLUE_NOISE_DIR, f"blue_noise_ref_masks_{suf}.npz")
    chol = os.path.join(BLUE_NOISE_DIR, f"blue_noise_chol_matrix_{suf}.pt")
    return masks, chol


def default_chol_path(dataset: str) -> str:
    """Cholesky path for config ``model.cov_save_path`` (repo-root relative)."""
    _, chol = blue_noise_filenames(dataset)
    return os.path.relpath(chol, _ROOT)


def get_rna_dimensions(file_path: str) -> tuple:
    """Infer (n_genes, n_timepoints) from cells×genes+h CSV."""
    df = pd.read_csv(file_path)
    genes = [c for c in df.columns if c != "h"]
    timepoints = sorted(df["h"].unique())
    return len(genes), len(timepoints)


def get_mesc_dimensions(file_path: str) -> tuple:
    """Infer (n_genes, n_timepoints) for mESC matrix (same caps as ``dataset_mesc``)."""
    from dataset_mesc import MAX_GENES, extract_time_and_id

    df = pd.read_csv(file_path, index_col=0)
    times = set()
    for col in df.columns:
        t, _ = extract_time_and_id(col)
        if t is not None:
            times.add(t)
    if not times:
        raise ValueError(f"No RamDA_mESC_* columns found in {file_path!r}")
    n_genes_total = len(df.index)
    n_genes = min(MAX_GENES, n_genes_total)
    return n_genes, len(sorted(times))


def precompute_radial_bins(nrows, ncols, nbins):
    fx = np.fft.fftfreq(ncols)
    fy = np.fft.fftfreq(nrows)
    fx, fy = np.meshgrid(fx, fy)
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftshift(fy)
    r = np.sqrt(fx**2 + fy**2)
    r_flat = r.flatten()
    r_bins = np.linspace(r_flat.min(), r_flat.max(), nbins + 1)
    bin_centers = np.zeros(nbins)
    for i in range(nbins):
        bin_centers[i] = (r_bins[i] + r_bins[i + 1]) / 2
    return r, r_bins, bin_centers


def compute_energy(mask, radial_r, r_bins, bin_centers, nbins):
    mask_f = mask.astype(float)
    fft2d = np.fft.fft2(mask_f)
    psd2d = np.abs(fft2d) ** 2
    psd2d_shifted = np.fft.fftshift(psd2d)

    r_flat = radial_r.flatten()
    psd_flat = psd2d_shifted.flatten()

    radial_profile = np.zeros(nbins)
    for i in range(nbins):
        bin_mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i + 1])
        if np.any(bin_mask):
            radial_profile[i] = np.mean(psd_flat[bin_mask])
        else:
            radial_profile[i] = 0.0

    def frequency_penalty(f):
        return np.abs(f) ** POWER_EXP + 0.1 / (np.abs(f) + 1e-4)

    ideal_spectrum = frequency_penalty(bin_centers)
    ideal_spectrum[0] = 0.0

    ideal_spectrum /= np.sum(ideal_spectrum)
    if np.sum(radial_profile) > 0:
        radial_profile /= np.sum(radial_profile)

    return np.mean((radial_profile - ideal_spectrum) ** 2)


def anneal_one_mask(n_genes, n_timepoints, radial_r, r_bins, bin_centers, nbins):
    total_elements = n_genes * n_timepoints
    num_ones = total_elements // 2
    mask_flat = np.zeros(total_elements, dtype=np.int32)
    mask_flat[:num_ones] = 1
    np.random.shuffle(mask_flat)
    mask = mask_flat.reshape(n_genes, n_timepoints)

    energy = compute_energy(mask, radial_r, r_bins, bin_centers, nbins)
    temperature = INIT_TEMP
    acceptance_rate = 0

    for step in range(NUM_ITERATIONS):
        num_swaps = 1
        ones_idx = np.argwhere(mask == 1)
        zeros_idx = np.argwhere(mask == 0)
        if len(ones_idx) == 0 or len(zeros_idx) == 0:
            break

        chosen_ones = ones_idx[np.random.choice(len(ones_idx), size=num_swaps, replace=False)]
        chosen_zeros = zeros_idx[np.random.choice(len(zeros_idx), size=num_swaps, replace=False)]

        for i in range(num_swaps):
            r1, c1 = chosen_ones[i]
            r2, c2 = chosen_zeros[i]
            mask[r1, c1], mask[r2, c2] = mask[r2, c2], mask[r1, c1]

        new_energy = compute_energy(mask, radial_r, r_bins, bin_centers, nbins)
        delta_e = new_energy - energy

        if (delta_e < 0) or (np.exp(-delta_e / temperature) > np.random.rand()):
            energy = new_energy
            acceptance_rate += 1
        else:
            for i in range(num_swaps):
                r1, c1 = chosen_ones[i]
                r2, c2 = chosen_zeros[i]
                mask[r1, c1], mask[r2, c2] = mask[r2, c2], mask[r1, c1]

        if step % ADAPTIVE_DECAY == 0 and acceptance_rate < 0.01 * ADAPTIVE_DECAY:
            temperature *= ANNEALING_RATE * 1.2
            acceptance_rate = 0
        else:
            temperature *= ANNEALING_RATE

        if temperature < FINAL_TEMP:
            break

    return mask


def build_blue_noise_cov(n_genes, n_timepoints, save_masks_path):
    radial_r, r_bins, bin_centers = precompute_radial_bins(n_genes, n_timepoints, NBINS)

    if not os.path.exists(save_masks_path):
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    anneal_one_mask,
                    n_genes,
                    n_timepoints,
                    radial_r,
                    r_bins,
                    bin_centers,
                    NBINS,
                )
                for _ in range(NUM_REF_MASKS)
            ]
            all_masks = []
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=NUM_REF_MASKS,
                desc="Annealing BN masks",
            ):
                try:
                    all_masks.append(fut.result())
                except Exception as exc:
                    print(f"An error occurred: {exc}")

        all_masks = np.stack(all_masks, axis=0)
        np.savez_compressed(save_masks_path, masks=all_masks)
    else:
        loaded = np.load(save_masks_path)
        all_masks = loaded["masks"]
        print(f"[INFO] Loaded {all_masks.shape[0]} reference BN masks from {save_masks_path}")

    n = all_masks.shape[0]
    flat_len = n_genes * n_timepoints
    data_flat = all_masks.reshape(n, flat_len).astype(np.float32)
    mean_flat = np.mean(data_flat, axis=0, keepdims=True)

    cov_mat = np.zeros((flat_len, flat_len), dtype=np.float64)
    for i in tqdm(range(n), desc="Computing Covariance Matrix"):
        diff = data_flat[i] - mean_flat
        cov_mat += np.outer(diff, diff)
    cov_mat /= n - 1

    return cov_mat.astype(np.float32)


def nearest_spd(a, num_iters=5):
    a_sym = 0.5 * (a + a.T)
    for _ in range(num_iters):
        w, v = np.linalg.eigh(a_sym)
        w_clamped = np.maximum(w, 1e-7)
        a_sym = (v * w_clamped) @ v.T
    return a_sym


def compute_and_save_chol(dataset: str, input_path: Optional[str] = None) -> None:
    if dataset == "rna":
        raw = input_path or DEFAULT_RNA_CSV
        csv_path = _resolve_csv_path(raw) if input_path else raw
        n_genes, n_timepoints = get_rna_dimensions(csv_path)
    else:
        from dataset_mesc import get_mesc_file

        raw = input_path or get_mesc_file()
        csv_path = _resolve_csv_path(raw)
        n_genes, n_timepoints = get_mesc_dimensions(csv_path)

    save_masks_path, save_chol_path = blue_noise_filenames(dataset)

    print(
        f"[gen_bn] dataset={dataset!r} | input={csv_path!r} | "
        f"n_genes={n_genes}, n_timepoints={n_timepoints} (mask shape matches CSDI tile_k × tile_l)"
    )
    print(f"[gen_bn] masks -> {save_masks_path}")
    print(f"[gen_bn] Cholesky -> {save_chol_path}")

    print("[INFO] Computing blue noise covariance matrix...")
    cov_mat = build_blue_noise_cov(n_genes, n_timepoints, save_masks_path)

    print("[INFO] Converting to nearest SPD matrix...")
    stable_cov = nearest_spd(cov_mat, num_iters=3)
    stable_cov += 1e-7 * np.eye(stable_cov.shape[0], dtype=stable_cov.dtype)

    try:
        with tqdm(total=1, desc="Computing Cholesky Decomposition") as pbar:
            l_chol = np.linalg.cholesky(stable_cov)
            pbar.update(1)
        l_torch = torch.from_numpy(l_chol).float()
        torch.save(l_torch, save_chol_path)
        print(f"[INFO] Saved Cholesky factor to {save_chol_path}")
        print(
            f"[gen_bn] Set model.cov_save_path to {default_chol_path(dataset)!r} "
            f"(or absolute path) when use_blue_noise is true."
        )
    except np.linalg.LinAlgError:
        print("[ERROR] Cholesky decomposition failed. Increase diagonal adjustment or refine SPD approach.")


def main():
    parser = argparse.ArgumentParser(description="Generate blue-noise reference masks and Cholesky factor.")
    parser.add_argument(
        "--dataset",
        choices=("rna", "mesc"),
        required=True,
        help="rna: cells×genes+h CSV; mESC: ExpressionData genes×samples matrix",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override default CSV (RNA: default rna_reordered_dpt.csv; mESC: data/mESC/ExpressionData.csv)",
    )
    args = parser.parse_args()
    compute_and_save_chol(args.dataset, args.input)


if __name__ == "__main__":
    main()
