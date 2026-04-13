import os
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.optimize import linear_sum_assignment
from diff_models import diff_CSDI


def _compute_cumulative_instant_blue(num_steps, beta, gamma_start, gamma_end, gamma_tau):
    """
    Same *target shape* as the sigmoid schedule: S(i)=σ(γ_s+(γ_e-γ_s)(i/T)^τ) with i=0..T-1
    (matches get_noise_blend_weight for schedule "sigmoid"). Interpret S as a cumulative
    progression and set G*(i)=(S(i)-S(0))/(S(T-1)-S(0)) in [0,1].

    Then N_0=0, N_{i+1}=(1-β_i)N_i+β_i and per-step linear blue fraction γ_i with the same
    mixing as sigmoid/linear: ε = (1-γ_i) ε_white + γ_i ε_blue (Gaussian weight w = 1-γ_i).

    Training and imputation both use the same precomputed γ_i at each timestep.
    """
    T = int(num_steps)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape[0] != T:
        raise ValueError(f"beta length {beta.shape[0]} != num_steps {T}")

    ti = np.arange(T, dtype=np.float64) / float(max(T, 1))
    x = float(gamma_start) + (float(gamma_end) - float(gamma_start)) * (ti ** float(gamma_tau))
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    den = float(s[-1] - s[0])
    if abs(den) < 1e-12:
        g_star = np.linspace(0.0, 1.0, T, dtype=np.float64)
    else:
        g_star = (s - s[0]) / den
    g_star = np.clip(g_star, 0.0, 1.0)

    n_arr = np.zeros(T + 1, dtype=np.float64)
    for i in range(T):
        n_arr[i + 1] = (1.0 - beta[i]) * n_arr[i] + beta[i]

    gamma = np.zeros(T, dtype=np.float64)
    for i in range(T):
        b = max(float(beta[i]), 1e-8)
        g_curr = g_star[i]
        g_prev = g_star[i - 1] if i > 0 else 0.0
        numer = g_curr * n_arr[i + 1] - (1.0 - beta[i]) * g_prev * n_arr[i]
        gamma[i] = numer / b
    return np.clip(gamma, 0.0, 1.0)


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, tile_k=None, tile_l=None):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        # 1) Basic model config
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        # side info dimension
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim
        )

        # 2) Diffusion Model
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"]**0.5,
                config_diff["beta_end"]**0.5,
                self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device)
            .unsqueeze(1).unsqueeze(1)
        )

        # 3) Blue vs white noise: correlated (gen_bn Cholesky) vs i.i.d. Gaussian (vanilla CSDI)
        self.use_blue_noise = config["model"].get("use_blue_noise", True)
        # RNA passes tile_k/tile_l from the dataset; Physio/Forecasting omit them and use defaults here.
        self.tile_k = tile_k if tile_k is not None else 100
        self.tile_l = tile_l if tile_l is not None else 5
        self.cov_save_path = config["model"].get(
            "cov_save_path", "blue_noise/blue_noise_chol_matrix_rna.pt"
        )
        if self.use_blue_noise:
            self.L_chol_small = self.create_covariance_matrix_2d(
                K=self.tile_k,
                L_max=self.tile_l,
                save_path=self.cov_save_path
            ).to(self.device)
        else:
            self.L_chol_small = None
            print(
                "[INFO] White noise mode (vanilla CSDI): use_blue_noise=false; "
                "no blue-noise Cholesky loaded."
            )

        # 4) noise blend (only used when use_blue_noise): w * gaussian + (1-w) * correlated blue noise
        _nbs = config["model"].get("noise_blend_schedule", "sigmoid")
        self.noise_blend_schedule = str(_nbs).lower().strip()
        _allowed_sched = ("sigmoid", "linear", "cumulative", "step")
        if self.noise_blend_schedule not in _allowed_sched:
            raise ValueError(
                f"model.noise_blend_schedule must be one of {_allowed_sched}, got {_nbs!r}"
            )
        self.gamma_start = config["model"].get("gamma_start", 0.0)
        self.gamma_end   = config["model"].get("gamma_end",   3.0)
        self.gamma_tau   = config["model"].get("gamma_tau",   0.2)
        _step_t = config["model"].get("noise_blend_step_t")
        if _step_t is None:
            _step_t = self.num_steps // 2
        self.step_blue_steps = int(max(0, min(int(_step_t), self.num_steps)))
        self.noise_blend_w_start = float(config["model"].get("noise_blend_w_start", 0.0))
        self.noise_blend_w_start = max(0.0, min(1.0, self.noise_blend_w_start))
        _per_rev = config["model"].get(f"noise_blend_reverse_{self.noise_blend_schedule}")
        if _per_rev is not None:
            self.noise_blend_reverse = bool(_per_rev)
        else:
            self.noise_blend_reverse = bool(config["model"].get("noise_blend_reverse", False))
        if self.noise_blend_schedule == "cumulative":
            cg = _compute_cumulative_instant_blue(
                self.num_steps,
                self.beta,
                self.gamma_start,
                self.gamma_end,
                self.gamma_tau,
            )
            self.register_buffer(
                "cumulative_instant_blue",
                torch.tensor(cg, dtype=torch.float32),
            )
            print(
                "[INFO] noise_blend_schedule=cumulative: G* = rescaled sigmoid curve; "
                "linear mix like sigmoid schedule; per-step γ from N_t recurrence"
            )
        else:
            self.register_buffer(
                "cumulative_instant_blue",
                torch.zeros(self.num_steps, dtype=torch.float32),
            )

        if self.noise_blend_schedule == "step":
            tb, Tn = self.step_blue_steps, self.num_steps
            w0 = self.noise_blend_w_start
            print(
                f"[INFO] noise_blend_schedule=step: w={w0} for i < {tb}, w=1 for i >= {tb}; num_steps={Tn}"
            )
        if self.noise_blend_schedule == "linear" and self.noise_blend_w_start != 0.0:
            print(
                f"[INFO] noise_blend_schedule=linear: w from {self.noise_blend_w_start} -> 1.0 over t "
                f"(noise_blend_w_start)"
            )
        if self.noise_blend_reverse:
            print(
                "[INFO] noise_blend_reverse=true: effective blend time (T-1)-t "
                f"(schedule={self.noise_blend_schedule})"
            )

        # 5) Standard rectified mapping using Hungarian
        self.use_rectified_mapping = config["model"].get("use_rectified_mapping", True)
        self.rectify_lambda = config["model"].get("rectify_lambda", 1.0)  # partial blend factor

    # ------------------------------------------------
    # Load blue noise Cholesky factor from file
    # ------------------------------------------------
    def create_covariance_matrix_2d(
        self, K, L_max, save_path="blue_noise/blue_noise_chol_matrix_rna.pt"
    ):
        path = save_path
        if not os.path.isabs(path) and not os.path.exists(path):
            _repo = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.join(_repo, path)
            if os.path.exists(alt):
                path = alt
        if os.path.exists(path):
            print(f"[INFO] Loading blue noise Cholesky factor from '{path}'")
            return torch.load(path)
        raise FileNotFoundError(
            f"Blue noise Cholesky matrix not found at '{save_path}' (cwd={os.getcwd()}). "
            "Generate it with: python gen_bn.py --dataset rna   or   --dataset mesc"
        )

    # ------------------------------------------------
    # Generate a single tile
    # ------------------------------------------------
    def generate_correlated_noise_tile(self, B):
        D_tile = self.tile_k * self.tile_l
        z = torch.randn(B, D_tile, device=self.device)
        # Use the blue noise Cholesky factor for correlation
        correlated = z @ self.L_chol_small[:D_tile, :D_tile].T
        return correlated.view(B, self.tile_k, self.tile_l)

    # ------------------------------------------------
    # Tiled approach if too big
    # ------------------------------------------------
    def generate_correlated_noise_2d_tiled(self, B, K, L):
        big_noise = torch.zeros(B, K, L, device=self.device)
        num_tiles_k = (K + self.tile_k - 1) // self.tile_k
        num_tiles_l = (L + self.tile_l - 1) // self.tile_l

        for i in range(num_tiles_k):
            for j in range(num_tiles_l):
                start_k = i * self.tile_k
                end_k = min(start_k + self.tile_k, K)
                start_l = j * self.tile_l
                end_l = min(start_l + self.tile_l, L)

                tile_noise = self.generate_correlated_noise_tile(B)
                cropped_tile = tile_noise[:, :end_k - start_k, :end_l - start_l]
                big_noise[:, start_k:end_k, start_l:end_l] = cropped_tile

        return big_noise

    def generate_correlated_noise_2d(self, B, K, L):
        if K <= self.tile_k and L <= self.tile_l:
            tile_noise = self.generate_correlated_noise_tile(B)
            return tile_noise[:, :K, :L]
        else:
            return self.generate_correlated_noise_2d_tiled(B, K, L)

    def sample_diffusion_noise(self, B, K, L, t):
        """
        Noise ε for the forward noising step.
        White: i.i.d. Gaussian. Blue: blend of Gaussian and Cholesky-correlated noise (schedule in get_noise_blend_weight).
        """
        noise_gauss = torch.randn(B, K, L, device=self.device)
        if not self.use_blue_noise:
            return noise_gauss
        noise_corr = self.generate_correlated_noise_2d(B, K, L)
        blend_w = self.get_noise_blend_weight(t).view(B, 1, 1)
        return blend_w * noise_gauss + (1.0 - blend_w) * noise_corr

    def _map_blend_time(self, t_tensor):
        """Forward: use t. Reverse: (T-1)-t so w(t)=w_fwd(T-1-t) on indices 0..T-1."""
        t = t_tensor.float()
        if not self.noise_blend_reverse:
            return t
        return (self.num_steps - 1) - t

    def get_gamma(self, t_tensor):
        """
        Power-warped sigmoid schedule (noise_blend_schedule: sigmoid).
        Gaussian weight w = σ(γ_s + (γ_e-γ_s)(t/T)^τ). Other schedules do not use this.
        """
        t = self._map_blend_time(t_tensor)
        x = self.gamma_start + (self.gamma_end - self.gamma_start) * (
            (t / self.num_steps) ** self.gamma_tau
        )
        return torch.sigmoid(x)

    def get_noise_blend_weight(self, t_tensor):
        """
        Weight w in [0, 1] on Gaussian noise vs correlated noise:
        ε = w ε_white + (1-w) ε_blue. Sigmoid uses get_gamma(); cumulative uses precomputed 1-γ_t;
        linear/step use noise_blend_w_start where applicable.
        """
        if self.noise_blend_schedule == "cumulative":
            idx = t_tensor.long().clamp(0, self.num_steps - 1)
            if self.noise_blend_reverse:
                idx = (self.num_steps - 1) - idx
            return 1.0 - self.cumulative_instant_blue[idx].float()
        if self.noise_blend_schedule == "linear":
            t = self._map_blend_time(t_tensor)
            denom = max(self.num_steps - 1, 1)
            u = (t / denom).clamp(0.0, 1.0)
            w0 = self.noise_blend_w_start
            return w0 + (1.0 - w0) * u
        if self.noise_blend_schedule == "step":
            idx = t_tensor.long().clamp(0, self.num_steps - 1)
            if self.noise_blend_reverse:
                idx = (self.num_steps - 1) - idx
            base = t_tensor.float()
            w0 = float(self.noise_blend_w_start)
            return torch.where(
                idx < self.step_blue_steps,
                torch.full_like(base, w0),
                torch.ones_like(base),
            )
        return self.get_gamma(t_tensor)

    # ------------------------------------------------
    # Standard rectification: Hungarian assignment
    # ------------------------------------------------
    def rectify_mapping(self, data_batch, noise_batch):
        B, K, L = data_batch.shape

        data_flat = data_batch.reshape(B, -1)
        noise_flat = noise_batch.reshape(B, -1)

        # cost => shape(B,B)
        cost = torch.cdist(data_flat, noise_flat, p=2.0)**2
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        # reorder noise => noise_batch[col_ind]
        hungarian_noise = noise_batch[col_ind]  # shape(B, K, L)

        if self.rectify_lambda >= 1.0:
            # full rectification
            return hungarian_noise
        else:
            # partial blend
            return self.rectify_lambda * hungarian_noise + (1.0 - self.rectify_lambda) * noise_batch

    # time embedding, etc.
    def time_embedding(self, pos, d_model=128):
        B, L = pos.shape
        pe = torch.zeros(B, L, d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1.0 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2, device=self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            topk_indices = rand_for_mask[i].topk(num_masked).indices
            rand_for_mask[i][topk_indices] = -1

        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            if self.target_strategy == "mix" and np.random.rand() > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    # calc_loss => time-based blend
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = torch.full((B,), float(set_t), device=self.device)
        else:
            t = torch.randint(0, self.num_steps, (B,), device=self.device).float()

        current_alpha = self.alpha_torch[t.long()]

        # 1) white (Gaussian) or blue (correlated + blend schedule)
        noise = self.sample_diffusion_noise(B, K, L, t)

        # 2) standard rectification
        if self.use_rectified_mapping and is_train == 1:
            noise = self.rectify_mapping(observed_data, noise)

        # 3) forward noising
        noisy_data = (current_alpha**0.5) * observed_data + (1.0 - current_alpha)**0.5 * noise

        t_int = t.long()
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t_int)

        # 4) loss
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss_t = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss_t.detach()
        return loss_sum / self.num_steps

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            return torch.cat([cond_obs, noisy_target], dim=1)

    # Imputation => reverse process
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                # Match training: same white vs blue rule
                t_batch = torch.full((B,), float(t), device=self.device, dtype=torch.float32)
                noise_blend = self.sample_diffusion_noise(B, K, L, t_batch)

                if self.use_rectified_mapping:
                    noise_blend = self.rectify_mapping(observed_data, noise_blend)

                if self.is_unconditional:
                    diff_input = cond_mask * observed_data + (1 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t], device=self.device))

                coeff1 = 1 / self.alpha_hat[t]**0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t])**0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    sigma = ((1.0 - self.alpha[t-1]) / (1.0 - self.alpha[t]) * self.beta[t])**0.5
                    current_sample += sigma * noise_blend

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):
                target_mask[i, ..., :cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
    
class CSDI_RNA(CSDI_base):
    """CSDI for RNA-style batches: (cells, timepoints, genes); blue-noise tile K×L from dataset."""

    def __init__(self, config, device, target_dim, num_timepoints=None):
        tile_k = target_dim
        tile_l = num_timepoints if num_timepoints is not None else 5
        super().__init__(target_dim, config, device, tile_k=tile_k, tile_l=tile_l)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
