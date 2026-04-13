# tBN-CSDI

Conditional diffusion for spatiotemporal RNA-style data (genes × experimental time per cell). This repo extends the usual CSDI setup with **optional correlated “blue” noise** (Cholesky-colored Gaussian on a 2D gene×time tile), a **time-dependent blend** between white and blue noise, and **rectified noise** (Hungarian-based alignment). You can run **vanilla CSDI** (white noise only) with a single flag.

---

## Setup

```bash
cd /path/to/tBN-CSDI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train and evaluate from the **repository root** so paths like `config/base.yaml` and `blue_noise/…` resolve correctly.

---

## Quick reference: what to run

| Goal | Command |
|------|---------|
| Precompute blue-noise Cholesky (RNA layout) | `python gen_bn.py --dataset rna` |
| Precompute blue-noise Cholesky (mESC layout) | `python gen_bn.py --dataset mesc` |
| Train on standard RNA CSV | `python exe_rna.py` |
| Train on mESC matrix | `python exe_mesc.py` |
| Train on pseudotime-reordered RNA | Generate CSV with `reorder_rna.py`, then `python exe_rna_reorder.py` |
| Train on pseudotime-reordered mESC | Generate CSV with `reorder_rna.py --dataset mesc`, then `python exe_mesc_reorder.py` |
| **Vanilla CSDI** (no blue noise) | Add `--white-noise` to any `exe_*.py` (or set `use_blue_noise: false` in YAML) |

---

## Generating blue noise (`gen_bn.py`)

Blue noise is defined on a **2D grid of size `(n_genes, n_timepoints)`**, matching the diffusion noise layout (`tile_k × tile_l`) for one training example. The script builds many binary masks (simulated annealing), estimates their covariance, projects to a positive semidefinite matrix, and saves a **Cholesky factor**.

**Outputs** (created under `blue_noise/`):

- `blue_noise_ref_masks_{rna|mesc}.npz` — large; listed in `.gitignore`
- `blue_noise_chol_matrix_{rna|mesc}.pt` — used at train time

**Commands:**

```bash
python gen_bn.py --dataset rna
python gen_bn.py --dataset rna --input data/rna/rna.csv          # match the CSV you train on
python gen_bn.py --dataset mesc
python gen_bn.py --dataset mesc --input data/mESC/ExpressionData.csv
```

**Defaults:**

- **RNA:** dimensions inferred from `data/rna/rna_reordered_dpt.csv` (override with `--input` to match your training file exactly).
- **mESC:** dimensions from `data/mESC/ExpressionData.csv` (same gene/time logic as `dataset_mesc.py`, including `MAX_GENES`).

**Important:** `(n_genes, n_timepoints)` must match the dataloader used in training. Reordering **rows** in a CSV does **not** change gene count or distinct `h` values, so the Cholesky size is unchanged—but training on a **different** CSV (different columns or timepoints) requires regenerating `gen_bn` for that layout.

**Config:** set `model.cov_save_path` in YAML, e.g.:

- RNA: `blue_noise/blue_noise_chol_matrix_rna.pt` (`config/base.yaml`)
- mESC: `blue_noise/blue_noise_chol_matrix_mesc.pt` (`config/base_mesc.yaml`)

Wrapper: `./run_gen_bn.sh --dataset rna` (forwards args to `gen_bn.py`).

---

## Vanilla CSDI vs tBN-CSDI

| Mode | How | Noise in forward process |
|------|-----|---------------------------|
| **Vanilla CSDI** | `--white-noise` on `exe_*.py`, or `use_blue_noise: false` in config | i.i.d. Gaussian only |
| **tBN-CSDI** | `use_blue_noise: true` (default in `base.yaml` and `base_mesc.yaml`) and valid `cov_save_path` | Blend of Gaussian and Cholesky-correlated “blue” noise (see below) |

If `use_blue_noise: true` but the Cholesky file is missing, training will raise a clear error—run `gen_bn.py` for the matching dataset (`--dataset rna` or `mesc`) or pass **`--white-noise`** for vanilla CSDI without a matrix.

---

## Noise blend schedule (tBN-CSDI only)

When `use_blue_noise: true`, the per-step noise is:

\[
\epsilon = w(t)\,\epsilon_{\text{white}} + (1 - w(t))\,\epsilon_{\text{blue}}
\]

where \(w(t)\) is the **Gaussian weight** (higher \(w\) → more white noise). Schedule is controlled by `model.noise_blend_schedule` and related keys in `config/*.yaml`, or overridden from the CLI (see `rna_experiment_sweep.add_sweep_arguments`).

### Schedules (`noise_blend_schedule`)

| Value | Meaning |
|-------|---------|
| `sigmoid` | \(w(t) = \sigma(\gamma_{\text{start}} + (\gamma_{\text{end}}-\gamma_{\text{start}})(t/T)^{\tau})\) with `gamma_start`, `gamma_end`, `gamma_tau` |
| `linear` | \(w\) increases linearly from `noise_blend_w_start` to 1 over diffusion indices |
| `cumulative` | Same “shape” idea as the sigmoid schedule for the cumulative curve; per-step mixing uses the internal recurrence documented in `main_model.py` |
| `step` | \(w = \text{noise\_blend\_w\_start}\) for \(t < \text{noise\_blend\_step\_t}\), else \(w = 1\) |

**Sigmoid schedule:** the default `gamma_start: 0` is **not** “no Gaussian noise” at the first step. At \(t=0\), \(w(0)=\sigma(\gamma_{\text{start}})\), so **`gamma_start = 0` ⇒ \(w(0)=\sigma(0)=0.5\)** — equal mix of white and blue. To push the start toward mostly blue (small \(w\)), use a **negative** `gamma_start` (e.g. `--gamma-start -6` or `--nearly-all-blue-start`, giving \(\sigma(-6)\approx 0.0025\) white at \(t=0\)).

CLI overrides:

```bash
--noise-blend-schedule sigmoid     # linear | cumulative | step
--gamma-start -6                   # sigmoid: more blue at start (σ(-6)≈0.0025 white at t=0)
--nearly-all-blue-start            # sets gamma_start = -6
--blend-start 0.2                  # linear / step: initial Gaussian weight in [0,1]
--noise-blend-step-t 25            # step: threshold index (see YAML default null → num_steps//2)
--noise-blend-reverse              # use effective time (T-1)-t so the schedule runs backward along indices
```

YAML keys: `gamma_start`, `gamma_end`, `gamma_tau`, `noise_blend_w_start`, `noise_blend_step_t`, `noise_blend_reverse`, optional per-schedule `noise_blend_reverse_*` overrides (e.g. `noise_blend_reverse_sigmoid`; commented in `base.yaml`).

---

## Training entry points

All training scripts share the same sweep driver: **`rna_experiment_sweep.py`** (`add_sweep_arguments`, `run_sweep`). Typical pattern:

- Multiple **missing ratios** (default `0.1 0.3 0.5 0.7 0.9`) × **`ntrials`** (default 5).
- Each trial uses seed `base_seed + trial_index` for splits and dataset cache; the **same** train/valid/test cell split is used across missing ratios for a fair comparison.

### Scripts

| Script | Data | Default config |
|--------|------|----------------|
| `exe_rna.py` | `data/rna/rna.csv` | `config/base.yaml` |
| `exe_rna_reorder.py` | Reordered RNA (see below) | `config/base.yaml` |
| `exe_mesc.py` | `data/mESC/ExpressionData.csv` via `dataset_mesc` | `config/base_mesc.yaml` |
| `exe_mesc_reorder.py` | Reordered mESC wide CSV | `config/base_mesc.yaml` |

### Common CLI arguments

```
--config base.yaml              # or base_mesc.yaml; path under config/
--device cuda:0
--seed 1
--missingratios 0.1 0.5 0.9
--testmissingratio 0.5          # single ratio only
--ntrials 5
--nsample 100                   # posterior samples for evaluation
--modelfolder <name>            # if set: skip train, load save/<name>/model.pth
--white-noise                   # vanilla CSDI
# plus noise-blend flags listed above
```

**Outputs:** under `./save/<save_prefix>_<run_id>/` with `config.json`, per missing-ratio folders, metrics (RMSE, MAE, CRPS, etc.).

---

## Pseudotime reordering (`reorder_rna.py`)

Produces a **cells × (genes + `h`)** CSV with the **same** experimental labels `h` per cell, but **rows sorted by inferred pseudotime**. Training then uses `dataset_rna.parse_rna_data` as usual; only **within–timepoint row order** changes.

### Command-line

```bash
python reorder_rna.py --dataset rna --method dpt
python reorder_rna.py --dataset mesc --method dpt --seed 42
python reorder_rna.py --dataset auto --method phate
python reorder_rna.py --input /path/to/custom.csv --output /path/to/out.csv --method slingshot
```

| Argument | Role |
|----------|------|
| `--dataset` | `rna` (default), `mesc`, or `auto` (first preset input that exists) |
| `--method` | `dpt` (default), `slingshot`, `phate` |
| `--input` / `--output` | Override CSV paths (format auto-detected from header) |
| `--seed` | RNG for DPT roots / PHATE (default 42) |

**Preset outputs** (see `reorder_datasets.py`):

- RNA: `data/rna/rna_reordered_{dpt,slingshot,phate}.csv`
- mESC: `data/mESC/mesc_reordered_{dpt,slingshot,phate}.csv`

### Choosing a method

- **dpt** — Scanpy diffusion pseudotime, **multi-root consensus** from the earliest experimental time (Python only; default).
- **slingshot** — R package **slingshot** on the **same PCA** as the neighbor graph; clusters = experimental time. Requires **rpy2** and R `slingshot`.
- **phate** — 1D **PHATE** on the same PCA; orientation fixed by Spearman correlation with experimental time. Requires **phate**.

All three share the same preprocessing: gene filtering, HVG (Seurat flavor), scaling, PCA, kNN (cosine), diffusion map for DPT.

**mESC:** expression **0** is treated as missing in the reorder pipeline (zeros → NaN, then mean imputation after gene count filtering). Wide RNA uses NaNs as missing.

### Training on reordered CSVs

After generating the CSV:

```bash
python exe_rna_reorder.py --pseudotime-method dpt
python exe_mesc_reorder.py --pseudotime-method phate
```

`--pseudotime-method` must match the CSV you created with `reorder_rna.py --method`.

---

## Important code locations

| Piece | Location |
|-------|----------|
| Diffusion + blue/white noise + blend + rectified mapping | `main_model.py` (`CSDI_RNA`, `sample_diffusion_noise`, …) |
| RNA wide CSV → tensors | `dataset_rna.py` |
| mESC genes×samples → tensors | `dataset_mesc.py` |
| Training loop, evaluation metrics | `utils.py` |
| Sweep + CLI flags | `rna_experiment_sweep.py` |
| YAML configs | `config/base.yaml`, `config/base_mesc.yaml` |
| Blue-noise precompute | `gen_bn.py` |
| Reorder pipeline | `reorder_rna.py`, `reorder_datasets.py` |

**Diffusion schedule** (β schedule, `num_steps`, etc.) lives under `diffusion:` in the YAML files, separate from the **noise blend** schedule.
