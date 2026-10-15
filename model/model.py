"""TACD set representation, capacity allocation, task alignment and reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


def _advance_initialization_rng(*widths: int) -> None:
    """Keep the established seed schedule without registering unused parameters."""
    for in_width, out_width in zip(widths, widths[1:]):
        weight = torch.empty((out_width, in_width))
        bias = torch.empty(out_width)
        nn.init.kaiming_uniform_(weight, a=sqrt(5))
        bound = 1 / sqrt(in_width)
        nn.init.uniform_(bias, -bound, bound)


@dataclass
class AutoencoderOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    predicted_count: torch.Tensor
    target: torch.Tensor
    decode_count: int


def _base_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


def _task_mlp(in_dim: int, hidden: int, out_dim: int, depth: int=2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


def _zeros_like_count(length: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros((length, dim))


def _pad_direct(target: torch.Tensor, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_n, prediction_n = (target.shape[0], prediction.shape[0])
    max_n = max(target_n, prediction_n)
    if max_n == 0:
        return (target.new_zeros((0, target.shape[-1])), prediction.new_zeros((0, prediction.shape[-1])))
    target_pad = _zeros_like_count(max_n, target.shape[-1], target)
    prediction_pad = _zeros_like_count(max_n, prediction.shape[-1], prediction)
    if target_n:
        target_pad[:target_n] = target
    if prediction_n:
        prediction_pad[:prediction_n] = prediction
    return (target_pad, prediction_pad)


def _pad_hungarian(target: torch.Tensor, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_n, prediction_n = (target.shape[0], prediction.shape[0])
    dim = target.shape[-1] if target.ndim == 2 else prediction.shape[-1]
    max_n = max(target_n, prediction_n)
    if max_n == 0:
        return (target.new_zeros((0, dim)), prediction.new_zeros((0, dim)))
    target_pad = _zeros_like_count(max_n, dim, target if target_n else prediction)
    prediction_pad = _zeros_like_count(max_n, dim, prediction if prediction_n else target)
    if target_n == 0:
        prediction_pad[:prediction_n] = prediction
        return (target_pad, prediction_pad)
    if prediction_n == 0:
        target_pad[:target_n] = target
        return (target_pad, prediction_pad)
    cost = torch.cdist(target, prediction, p=2).pow(2).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    matched_target_rows = set()
    matched_pred_cols = set()
    slot = 0
    for row, col in zip(rows.tolist(), cols.tolist()):
        target_pad[slot] = target[row]
        prediction_pad[slot] = prediction[col]
        matched_target_rows.add(row)
        matched_pred_cols.add(col)
        slot += 1
    for row in range(target_n):
        if row not in matched_target_rows:
            target_pad[slot] = target[row]
            slot += 1
    for col in range(prediction_n):
        if col not in matched_pred_cols:
            prediction_pad[slot] = prediction[col]
            slot += 1
    return (target_pad, prediction_pad)


def align_sets(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and align two variable-cardinality sets with the requested policy."""
    if matching == 'direct':
        return _pad_direct(target, prediction)
    if matching == 'hungarian':
        return _pad_hungarian(target, prediction)
    raise ValueError(f'Unsupported matching mode: {matching}')


def mse_between_sets(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> torch.Tensor:
    """Return mean squared error after direct or Hungarian set alignment."""
    aligned_target, aligned_prediction = align_sets(target, prediction, matching)
    if aligned_target.numel() == 0:
        return aligned_target.new_tensor(0.0)
    return torch.mean((aligned_prediction - aligned_target) ** 2)


def set_mmd(x: torch.Tensor, y: torch.Tensor, bandwidths: tuple[float, ...]=(0.5, 1.0, 2.0, 4.0), weights: torch.Tensor | None=None) -> torch.Tensor:
    """Compute a weighted multi-bandwidth kernel MMD between two sets."""
    if x.shape[0] == 0 and y.shape[0] == 0:
        return x.new_tensor(0.0)

    def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[0] == 0 or b.shape[0] == 0:
            return x.new_zeros((a.shape[0], b.shape[0]))
        sqdist = torch.cdist(a, b, p=2.0).pow(2)
        if weights is None:
            local_weights = x.new_ones(len(bandwidths))
        else:
            local_weights = weights.to(device=x.device, dtype=x.dtype)
        kernels = [local_weights[idx] * torch.exp(-sqdist / (2.0 * bandwidth * bandwidth)) for idx, bandwidth in enumerate(bandwidths)]
        return sum(kernels) / local_weights.sum().clamp_min(1e-06)
    k_xx = _kernel(x, x).mean() if x.shape[0] > 0 else x.new_tensor(0.0)
    k_yy = _kernel(y, y).mean() if y.shape[0] > 0 else y.new_tensor(0.0)
    k_xy = _kernel(x, y).mean() if x.shape[0] > 0 and y.shape[0] > 0 else x.new_tensor(0.0)
    return k_xx + k_yy - 2.0 * k_xy


def moment_alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Match first and second moments of two variable-cardinality sets."""
    if x.shape[0] == 0 and y.shape[0] == 0:
        return x.new_tensor(0.0)
    if x.shape[0] == 0 or y.shape[0] == 0:
        base = x if x.shape[0] > 0 else y
        return base.new_tensor(1.0)
    mean_loss = (x.mean(dim=0) - y.mean(dim=0)).pow(2).mean()
    if x.shape[0] <= 1 or y.shape[0] <= 1:
        return mean_loss
    x_cov = torch.cov(x.transpose(0, 1))
    y_cov = torch.cov(y.transpose(0, 1))
    cov_loss = (x_cov - y_cov).pow(2).mean()
    return mean_loss + 0.25 * cov_loss


def scaled_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Preserve the forward value while scaling its backward gradient."""
    return x.detach() + scale * (x - x.detach())


def info_nce_loss(query: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute a single-query InfoNCE objective against queued negatives."""
    q = query / query.norm(p=2).clamp_min(1e-06)
    p = positive / positive.norm(p=2).clamp_min(1e-06)
    pos_logit = (torch.sum(q * p, dim=-1, keepdim=True) / temperature).view(1, 1)
    if negatives.numel() == 0:
        return -pos_logit.squeeze()
    neg = negatives / negatives.norm(dim=-1, keepdim=True).clamp_min(1e-06)
    neg_logits = (torch.matmul(q.unsqueeze(0), neg.transpose(0, 1)) / temperature).view(1, -1)
    logits = torch.cat([pos_logit, neg_logits], dim=-1)
    labels = torch.zeros(1, dtype=torch.long, device=query.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def should_advance_schedule(training: bool) -> bool:
    """Advance the shared schedule at each model forward."""
    return True


class SetAutoencoderBackbone(nn.Module):

    def __init__(self, input_dim: int=6, hidden_dim: int=128, latent_dim: int=96, max_n: int=16, use_rho: bool=True, encoder_variant: str='key_value'):
        super().__init__()
        if encoder_variant not in {'key_value', 'deepset'}:
            raise ValueError(f'Unsupported encoder variant: {encoder_variant}')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_n = max_n
        self.use_rho = use_rho
        self.encoder_variant = encoder_variant
        projection = torch.randn(input_dim)
        projection = projection / projection.norm()
        self.register_buffer('projection', projection)
        self.key_embedding = nn.Embedding(max_n + 1, latent_dim)
        self.value_net = _mlp(input_dim, hidden_dim, latent_dim)
        self.size_encoder = nn.Linear(1, latent_dim, bias=False)
        self.deepset_phi = _mlp(input_dim, hidden_dim, latent_dim)
        self.deepset_post = _mlp(latent_dim, hidden_dim, latent_dim)
        self.decoder = _mlp(latent_dim, hidden_dim, input_dim)
        self.size_predictor = _mlp(latent_dim, hidden_dim, 1)
        self.execution_stage = 'train'
        self.training_max_agents = 12

    def canonical_order(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.arange(x.shape[0], device=x.device)
        if not self.use_rho:
            return torch.arange(x.shape[0], device=x.device)
        scores = x @ self.projection
        return torch.argsort(scores)

    def canonicalize(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.canonical_order(x)]

    def _key_indices(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.arange(1, n + 1, device=device, dtype=torch.long)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.execution_stage == 'evaluation' and x.shape[0] > self.training_max_agents:
            order = self.canonical_order(x)
            target = x[order]
            n = target.shape[0]
            keys = self._element_queries(n, target.dtype, target.device)
            values = self.value_net(target)
            pooled = (keys * values).sum(dim=0) * (float(self.training_max_agents) / float(n))
            size = self.size_encoder(target.new_tensor([[float(self.training_max_agents)]])).squeeze(0)
            return (pooled + size, target, order)
        order = self.canonical_order(x)
        target = x[order]
        n = target.shape[0]
        size_term = self.size_encoder(torch.tensor([[float(n)]], dtype=target.dtype, device=target.device)).squeeze(0)
        if n == 0:
            return (size_term, target, order)
        if self.encoder_variant == 'key_value':
            key_indices = self._key_indices(n, target.device)
            keys = self.key_embedding(key_indices)
            values = self.value_net(target)
            latent = (keys * values).sum(dim=0)
        else:
            features = self.deepset_phi(target)
            pooled = features.sum(dim=0, keepdim=True)
            latent = self.deepset_post(pooled).squeeze(0)
        return (latent + size_term, target, order)

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        if self.execution_stage == 'evaluation' and n > self.training_max_agents:
            queries = self._element_queries(n, z.dtype, z.device)
            return self.decoder(z.unsqueeze(0) * queries)
        if n <= 0:
            return z.new_zeros((0, self.input_dim))
        key_indices = self._key_indices(n, z.device)
        queries = self.key_embedding(key_indices)
        element_states = z.unsqueeze(0) * queries
        return self.decoder(element_states)

    def predict_count(self, z: torch.Tensor) -> torch.Tensor:
        raw_count = self.size_predictor(z.unsqueeze(0)).squeeze(0)
        return torch.sigmoid(raw_count).squeeze(-1) * float(self.max_n)

    def infer_count(self, z: torch.Tensor) -> int:
        prediction = float(self.predict_count(z).item())
        return int(max(0, min(self.max_n, round(prediction))))

    def forward(self, x: torch.Tensor, decode_count: int | None=None, decode_mode: str='predicted') -> AutoencoderOutput:
        latent, target, _ = self.encode(x)
        predicted_count = self.predict_count(latent)
        if decode_count is None:
            if decode_mode == 'predicted':
                decode_count = self.infer_count(latent)
            elif decode_mode == 'target':
                decode_count = int(target.shape[0])
            else:
                raise ValueError(f'Unsupported decode_mode: {decode_mode}')
        reconstruction = self.decode(latent, decode_count)
        return AutoencoderOutput(reconstruction=reconstruction, latent=latent, predicted_count=predicted_count, target=target, decode_count=int(decode_count))

    def set_execution_stage(self, stage: str):
        if stage not in {'train', 'evaluation'}:
            raise ValueError(f'Unsupported execution stage: {stage}')
        self.execution_stage = stage
        return self

    def _element_queries(self, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        position = torch.linspace(1.0, float(self.training_max_agents), steps=n, dtype=dtype, device=device)
        low = position.floor().to(torch.long)
        high = position.ceil().to(torch.long)
        fraction = (position - low.to(dtype)).unsqueeze(-1)
        low_value = self.key_embedding(low)
        high_value = self.key_embedding(high)
        return low_value + fraction * (high_value - low_value)


class BranchSetAutoencoder(torch.nn.Module):

    def __init__(self, input_dim: int=6, hidden_dim: int=128, latent_dim: int=96, num_slots: int=4, slot_dim: int=16, semantic_dim: int=32, max_n: int=16, total_steps_hint: int=160, semantic_mode: str='off', semantic_positive_style: str='noise', semantic_temperature: float=0.25, semantic_grad_scale: float=0.0, semantic_queue_size: int=32, semantic_start_frac: float=0.0, semantic_warmup_frac: float=0.0, task_condition_mode: str='off', task_condition_scale: float=0.0, coarse_warmup_frac: float=0.0, coarse_supervision_scale: float=0.0, coarse_supervision_frac: float=0.0, coarse_supervision_floor: float=0.0, complexity_level_floor_alpha: float=0.0, complexity_level_floor_power: float=1.0, dynamic_similarity_scale: float=0.0, dynamic_similarity_mode: str='simple', dynamic_gate_floor: float=0.0, dynamic_gate_ceiling: float=1.0, dynamic_gate_reg_scale: float=0.0, count_head_mode: str='shared', kernel_weight_mode: str='uniform', kernel_prior_mix: float=1.0, kernel_prior_scale: float=0.0, anchor_mode: str='off', anchor_scale: float=0.0, distribution_ramp_denom: float=0.0, hysteresis_scale: float=0.0, hysteresis_budget_scale: float=0.0, dist_scale: float=1.0, moment_scale: float=0.0, budget_scale: float=0.11, budget_floor_alpha: float=0.0, budget_lower_bound_alpha: float=0.0, budget_lower_bound_power: float=1.0, budget_lower_bound_scale: float=0.0, slot_gate_floor_alpha: float=0.0, slot_gate_floor_scale: float=0.6, entropy_scale: float=0.08, active_scale: float=0.1, ordering_scale: float=0.05, diversity_scale: float=0.04, complexity_scale: float=0.07, semantic_scale: float=0.0, subset_scale: float=0.0, negative_scale: float=0.0, margin_scale: float=0.0, aux_scale: float=0.0, monotonicity_scale: float=0.07, improvement_scale: float=0.2, residual_scale: float=0.03, slot_aggregation_mode: str='mean', mixture_temperature: float=1.0):
        super().__init__()
        self.backbone = SetAutoencoderBackbone(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, max_n=max_n)
        self.coarse_dim = latent_dim // 2
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.max_n = max_n
        self.total_steps_hint = max(1, int(total_steps_hint))
        self.semantic_mode = semantic_mode
        self.semantic_positive_style = semantic_positive_style
        self.semantic_temperature = semantic_temperature
        self.semantic_grad_scale = semantic_grad_scale
        self.semantic_start_frac = semantic_start_frac
        self.semantic_warmup_frac = semantic_warmup_frac
        self.task_condition_mode = task_condition_mode
        self.task_condition_scale = task_condition_scale
        self.coarse_warmup_frac = coarse_warmup_frac
        self.coarse_supervision_scale = coarse_supervision_scale
        self.coarse_supervision_frac = coarse_supervision_frac
        self.coarse_supervision_floor = coarse_supervision_floor
        self.complexity_level_floor_alpha = complexity_level_floor_alpha
        self.complexity_level_floor_power = complexity_level_floor_power
        self.dynamic_similarity_scale = dynamic_similarity_scale
        self.dynamic_similarity_mode = dynamic_similarity_mode
        self.dynamic_gate_floor = dynamic_gate_floor
        self.dynamic_gate_ceiling = dynamic_gate_ceiling
        self.dynamic_gate_reg_scale = dynamic_gate_reg_scale
        self.count_head_mode = count_head_mode
        self.kernel_weight_mode = kernel_weight_mode
        self.kernel_prior_mix = kernel_prior_mix
        self.kernel_prior_scale = kernel_prior_scale
        self.anchor_mode = anchor_mode
        self.anchor_scale = anchor_scale
        self.distribution_ramp_denom = distribution_ramp_denom
        self.hysteresis_scale = hysteresis_scale
        self.hysteresis_budget_scale = hysteresis_budget_scale
        self.dist_scale = dist_scale
        self.moment_scale = moment_scale
        self.budget_scale = budget_scale
        self.budget_floor_alpha = budget_floor_alpha
        self.budget_lower_bound_alpha = budget_lower_bound_alpha
        self.budget_lower_bound_power = budget_lower_bound_power
        self.budget_lower_bound_scale = budget_lower_bound_scale
        self.slot_gate_floor_alpha = slot_gate_floor_alpha
        self.slot_gate_floor_scale = slot_gate_floor_scale
        self.entropy_scale = entropy_scale
        self.active_scale = active_scale
        self.ordering_scale = ordering_scale
        self.diversity_scale = diversity_scale
        self.complexity_scale = complexity_scale
        self.semantic_scale = semantic_scale
        self.subset_scale = subset_scale
        self.negative_scale = negative_scale
        self.margin_scale = margin_scale
        self.aux_scale = aux_scale
        self.monotonicity_scale = monotonicity_scale
        self.improvement_scale = improvement_scale
        self.residual_scale = residual_scale
        self.slot_aggregation_mode = slot_aggregation_mode
        self.mixture_temperature = mixture_temperature
        self.coarse_mu = _base_mlp(latent_dim, hidden_dim, self.coarse_dim)
        self.coarse_logvar = _base_mlp(latent_dim, hidden_dim, self.coarse_dim)
        self.slot_mu = _base_mlp(latent_dim, hidden_dim, num_slots * slot_dim)
        self.slot_logvar = _base_mlp(latent_dim, hidden_dim, num_slots * slot_dim)
        self.slot_logits = _base_mlp(latent_dim, hidden_dim, num_slots)
        self.count_hint_head = torch.nn.Linear(latent_dim, 1)
        self.complexity_head = torch.nn.Linear(latent_dim, 1)
        self.coarse_to_latent = _base_mlp(self.coarse_dim, hidden_dim, latent_dim)
        self.slot_summary_to_latent = _base_mlp(slot_dim, hidden_dim, latent_dim)
        self.query_embedding = torch.nn.Embedding(max_n + 1, slot_dim)
        self.coarse_refiner = _base_mlp(input_dim + latent_dim, hidden_dim, input_dim)
        self.residual_decoder = _base_mlp(input_dim + slot_dim + latent_dim, hidden_dim, input_dim)
        self.count_regressor = _base_mlp(latent_dim + num_slots, hidden_dim, 1)
        self.task_bins = 4
        self.semantic_input_dim = latent_dim * 2 + num_slots + 2 + self.task_bins
        self.semantic_input_norm = torch.nn.LayerNorm(self.semantic_input_dim)
        self.semantic_projector = torch.nn.Sequential(torch.nn.Linear(self.semantic_input_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, semantic_dim))
        self.semantic_predictor = torch.nn.Sequential(torch.nn.Linear(semantic_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, semantic_dim))
        self.semantic_aux_head = _base_mlp(semantic_dim, hidden_dim, 3)
        self.task_embedding = torch.nn.Embedding(self.task_bins, latent_dim)
        _advance_initialization_rng(latent_dim, hidden_dim, hidden_dim, self.task_bins)
        self.similarity_gate = torch.nn.Sequential(torch.nn.Linear(6, hidden_dim // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_dim // 2, 1), torch.nn.Sigmoid())
        self.slot_aggregator = torch.nn.Sequential(torch.nn.Linear(slot_dim + 4, hidden_dim // 2), torch.nn.LayerNorm(hidden_dim // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_dim // 2, 1))
        self.kernel_mixer = torch.nn.Sequential(torch.nn.Linear(6, hidden_dim // 2), torch.nn.LayerNorm(hidden_dim // 2), torch.nn.ReLU(), torch.nn.Linear(hidden_dim // 2, 4))
        self.refine_gain = torch.nn.Parameter(torch.tensor(0.3))
        self.coarse_refine_gain = torch.nn.Parameter(torch.tensor(0.18))
        self.residual_gain = torch.nn.Parameter(torch.tensor(0.22))
        self.register_buffer('slot_positions', torch.linspace(0.12, 0.88, steps=num_slots))
        self.register_buffer('semantic_queue', torch.zeros(semantic_queue_size, semantic_dim))
        self.register_buffer('semantic_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('semantic_queue_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('train_step', torch.zeros(1, dtype=torch.long))
        self.count_calibrator = _base_mlp(latent_dim + num_slots + 2, hidden_dim, 1)
        self.last_aux: dict[str, object] = {}
        self.last_aux_target: torch.Tensor | None = None
        self.last_aux_reconstruction: torch.Tensor | None = None

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _semantic_phase_weight(self) -> float:
        progress = min(1.0, float(self.train_step.item()) / float(self.total_steps_hint))
        if progress < self.semantic_start_frac:
            return 0.0
        if self.semantic_warmup_frac <= self.semantic_start_frac:
            return 1.0
        warm = (progress - self.semantic_start_frac) / (self.semantic_warmup_frac - self.semantic_start_frac)
        return max(0.0, min(1.0, warm))

    def _detail_phase_weight(self) -> float:
        progress = min(1.0, float(self.train_step.item()) / float(self.total_steps_hint))
        if progress < self.coarse_warmup_frac:
            return max(0.0, min(1.0, progress / max(self.coarse_warmup_frac, 1e-06)))
        return 1.0

    def _coarse_supervision_weight(self) -> float:
        if self.coarse_supervision_scale <= 0.0:
            return 0.0
        progress = min(1.0, float(self.train_step.item()) / float(self.total_steps_hint))
        if self.coarse_supervision_frac <= 0.0:
            return max(0.0, float(self.coarse_supervision_floor))
        if progress >= self.coarse_supervision_frac:
            return max(0.0, float(self.coarse_supervision_floor))
        decay = 1.0 - progress / max(self.coarse_supervision_frac, 1e-06)
        return max(float(self.coarse_supervision_floor), decay)

    def _augment_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        perm = torch.randperm(x.shape[0], device=x.device)
        if self.semantic_positive_style == 'noise':
            base = x[perm]
            noisy = base + 0.02 * torch.randn_like(base)
            scale = 0.94 + 0.12 * torch.rand((1, x.shape[1]), device=x.device, dtype=x.dtype)
            return noisy * scale
        if self.semantic_positive_style == 'subset_noise':
            keep = x.shape[0]
            if x.shape[0] >= 4:
                keep_ratio = 0.65 + 0.3 * torch.rand((), device=x.device, dtype=x.dtype)
                keep = max(1, min(x.shape[0], int(round(float(x.shape[0]) * float(keep_ratio.item())))))
            base = x[perm[:keep]]
            noisy = base + 0.05 * torch.randn_like(base)
            scale = 0.82 + 0.36 * torch.rand((1, x.shape[1]), device=x.device, dtype=x.dtype)
            noisy = noisy * scale
            feature_keep = torch.rand_like(noisy) > 0.2
            return torch.where(feature_keep, noisy, 0.55 * noisy)
        raise ValueError(f'Unsupported semantic_positive_style: {self.semantic_positive_style}')

    def _subset_view(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return x
        keep = max(1, x.shape[0] - max(1, (x.shape[0] + 1) // 3))
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm[:keep]]

    def _subset_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] <= 1:
            return (x, x.new_zeros((0, x.shape[1])))
        keep = max(1, x.shape[0] - max(1, (x.shape[0] + 1) // 3))
        perm = torch.randperm(x.shape[0], device=x.device)
        subset_idx = perm[:keep]
        removed_idx = perm[keep:]
        return (x[subset_idx], x[removed_idx])

    def _negative_view(self, x: torch.Tensor) -> torch.Tensor:
        n_neg = int(torch.randint(low=0, high=self.max_n + 1, size=(1,), device=x.device).item())
        if n_neg == 0:
            return x.new_zeros((0, self.backbone.input_dim))
        shifted = torch.randn(n_neg, self.backbone.input_dim, device=x.device, dtype=x.dtype)
        if x.shape[0] > 0:
            shifted = shifted + 0.55 * x.mean(dim=0, keepdim=True)
        return shifted

    def _synthetic_task_id(self, complexity_stats: dict[str, torch.Tensor]) -> torch.Tensor:
        device = complexity_stats['size_fraction'].device
        size_fraction = float(complexity_stats['size_fraction'].item())
        spread = float(complexity_stats['spread'].item())
        crowding = float(complexity_stats['crowding'].item())
        if size_fraction < 0.35 and crowding < 0.35:
            task_id = 0
        elif size_fraction < 0.35:
            task_id = 1
        elif spread >= 0.45:
            task_id = 2
        else:
            task_id = 3
        return torch.tensor(task_id, device=device, dtype=torch.long)

    def _heuristic_kernel_weights(self, size_fraction: torch.Tensor, spread: torch.Tensor, crowding: torch.Tensor) -> torch.Tensor:
        local = (1.25 - 0.55 * size_fraction + 0.1 * spread).clamp(0.2, 1.5)
        medium = (0.9 + 0.2 * spread + 0.1 * crowding).clamp(0.2, 1.6)
        broad = (0.75 + 0.55 * size_fraction + 0.2 * spread).clamp(0.2, 1.8)
        global_scale = (0.6 + 0.9 * size_fraction + 0.35 * crowding).clamp(0.2, 2.0)
        return torch.stack([local, medium, broad, global_scale])

    def _kernel_weights(self, size_fraction: torch.Tensor, spread: torch.Tensor, crowding: torch.Tensor) -> torch.Tensor:
        if self.kernel_weight_mode == 'uniform':
            return size_fraction.new_ones(4)
        prior = self._heuristic_kernel_weights(size_fraction, spread, crowding)
        if self.kernel_weight_mode == 'adaptive':
            return prior
        if self.kernel_weight_mode == 'learned_adaptive':
            mixer_features = torch.stack([size_fraction, spread, crowding, size_fraction * spread, size_fraction * crowding, spread * crowding])
            learned_logits = self.kernel_mixer(mixer_features.unsqueeze(0)).squeeze(0)
            learned = 0.25 + 3.0 * torch.softmax(learned_logits, dim=-1)
            blended = self.kernel_prior_mix * prior + (1.0 - self.kernel_prior_mix) * learned
            return blended.clamp(0.2, 2.2)
        raise ValueError(f'Unsupported kernel_weight_mode: {self.kernel_weight_mode}')

    def _aggregate_slots(self, z_slots: torch.Tensor, slot_gates: torch.Tensor, complexity_level: torch.Tensor, size_fraction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted_slots = slot_gates.unsqueeze(-1) * z_slots
        if self.slot_aggregation_mode == 'mean':
            normalized = slot_gates / slot_gates.sum().clamp_min(1e-06)
            return (weighted_slots.mean(dim=0), normalized, z_slots.new_tensor(0.0))
        if self.slot_aggregation_mode == 'mixture_softmax':
            aggregator_features = torch.cat([z_slots, slot_gates.unsqueeze(-1), self.slot_positions.unsqueeze(-1), complexity_level.expand(self.num_slots, 1), size_fraction.expand(self.num_slots, 1)], dim=-1)
            logits = self.slot_aggregator(aggregator_features).squeeze(-1)
            logits = logits + torch.log(slot_gates.clamp_min(1e-05))
            mixture_weights = torch.softmax(logits / max(self.mixture_temperature, 0.0001), dim=0)
            mixture_entropy = -torch.sum(mixture_weights * torch.log(mixture_weights.clamp_min(1e-08))) / torch.log(z_slots.new_tensor(float(self.num_slots)))
            mean_summary = weighted_slots.mean(dim=0)
            selective_summary = torch.sum(mixture_weights.unsqueeze(-1) * weighted_slots, dim=0)
            detail_summary = 0.45 * mean_summary + 0.55 * selective_summary
            return (detail_summary, mixture_weights, mixture_entropy)
        raise ValueError(f'Unsupported slot_aggregation_mode: {self.slot_aggregation_mode}')

    def _anchor_groups(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.anchor_mode == 'off':
            return [x]
        if x.shape[0] == 0:
            return [x]
        if self.anchor_mode == 'sign_binary':
            pos = x[x[:, 0] >= 0]
            neg = x[x[:, 0] < 0]
            return [pos, neg]
        if self.anchor_mode == 'quadrant':
            groups = []
            for sx in (False, True):
                for sy in (False, True):
                    mask = x[:, 0] >= 0 if sx else x[:, 0] < 0
                    mask = mask & (x[:, 1] >= 0 if sy else x[:, 1] < 0)
                    groups.append(x[mask])
            return groups
        raise ValueError(f'Unsupported anchor_mode: {self.anchor_mode}')

    def _conditional_anchor_mmd(self, target: torch.Tensor, reconstruction: torch.Tensor, kernel_weights: torch.Tensor) -> torch.Tensor:
        target_groups = self._anchor_groups(target)
        recon_groups = self._anchor_groups(reconstruction)
        losses = []
        for target_group, recon_group in zip(target_groups, recon_groups):
            losses.append(set_mmd(target_group, recon_group, weights=kernel_weights))
        return sum(losses) / float(len(losses))

    def _get_semantic_queue(self) -> torch.Tensor:
        count = int(self.semantic_queue_count.item())
        if count <= 0:
            return self.semantic_queue[:0]
        return self.semantic_queue[:count]

    def _enqueue_semantic(self, embeddings: torch.Tensor) -> None:
        if embeddings.numel() == 0:
            return
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        queue_size = self.semantic_queue.shape[0]
        ptr = int(self.semantic_queue_ptr.item())
        count = int(self.semantic_queue_count.item())
        with torch.no_grad():
            for emb in embeddings:
                self.semantic_queue[ptr].copy_(emb)
                ptr = (ptr + 1) % queue_size
                count = min(queue_size, count + 1)
            self.semantic_queue_ptr[0] = ptr
            self.semantic_queue_count[0] = count

    def _set_complexity(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = x.shape[0]
        size_fraction = x.new_tensor(float(n) / float(self.max_n))
        if n <= 1:
            zero = x.new_tensor(0.0)
            return {'size_fraction': size_fraction, 'spread': zero, 'crowding': zero, 'complexity': 0.55 * size_fraction}
        centered = x - x.mean(dim=0, keepdim=True)
        spread = centered.pow(2).mean().sqrt()
        spread_score = torch.sigmoid(1.15 * (spread - 0.95))
        pairwise = torch.cdist(x, x)
        inf = torch.full_like(pairwise, float('inf'))
        nearest = torch.where(torch.eye(n, device=x.device, dtype=torch.bool), inf, pairwise).min(dim=1).values
        crowding = torch.exp(-nearest.mean())
        complexity = (0.48 * size_fraction + 0.27 * spread_score + 0.25 * crowding).clamp(0.0, 1.0)
        return {'size_fraction': size_fraction, 'spread': spread_score, 'crowding': crowding, 'complexity': complexity}

    def _encode_state(self, x: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        base_latent, target, _ = self.backbone.encode(x)
        complexity_stats = self._set_complexity(target)
        task_id = self._synthetic_task_id(complexity_stats)
        task_one_hot = torch.nn.functional.one_hot(task_id, num_classes=self.task_bins).to(base_latent.dtype)
        if self.task_condition_mode == 'off':
            task_one_hot = torch.zeros_like(task_one_hot)
        coarse_mu = self.coarse_mu(base_latent)
        coarse_logvar = self.coarse_logvar(base_latent).clamp(-6.0, 2.0)
        slot_mu = self.slot_mu(base_latent).view(self.num_slots, self.slot_dim)
        slot_logvar = self.slot_logvar(base_latent).view(self.num_slots, self.slot_dim).clamp(-6.0, 2.0)
        slot_logits = self.slot_logits(base_latent).view(self.num_slots)
        count_hint = torch.sigmoid(self.count_hint_head(base_latent).squeeze(0))
        complexity_hint = torch.sigmoid(self.complexity_head(base_latent).squeeze(0))
        raw_complexity_level = (0.32 * count_hint + 0.28 * complexity_hint + 0.24 * complexity_stats['size_fraction'] + 0.16 * complexity_stats['crowding']).clamp(0.0, 1.0)
        complexity_level_floor = self.complexity_level_floor_alpha * complexity_stats['size_fraction'].pow(self.complexity_level_floor_power)
        complexity_level = torch.maximum(raw_complexity_level, complexity_level_floor).clamp(0.0, 1.0)
        raw_slot_gates = torch.sigmoid(slot_logits + 8.5 * (complexity_level - self.slot_positions))
        size_prior = torch.sigmoid(7.5 * (complexity_level - 0.18))
        slot_gates = (raw_slot_gates * (0.04 + 0.96 * size_prior)).clamp(0.0, 1.0)
        if self.slot_gate_floor_alpha > 0.0:
            slot_floor = self.slot_gate_floor_alpha * torch.sigmoid(10.0 * (complexity_stats['size_fraction'] - self.slot_gate_floor_scale * self.slot_positions))
            slot_gates = torch.maximum(slot_gates, slot_floor)
        else:
            slot_floor = torch.zeros_like(slot_gates)
        z_coarse = self._sample(coarse_mu, coarse_logvar) if sample else coarse_mu
        z_slots = self._sample(slot_mu, slot_logvar) if sample else slot_mu
        slot_budget = slot_gates.mean()
        detail_summary, mixture_weights, mixture_entropy = self._aggregate_slots(z_slots, slot_gates, complexity_level.unsqueeze(0), complexity_stats['size_fraction'].unsqueeze(0))
        coarse_latent = self.coarse_to_latent(z_coarse)
        detail_latent = self.slot_summary_to_latent(detail_summary)
        task_embedding = self.task_embedding(task_id)
        if self.task_condition_mode != 'off':
            coarse_latent = coarse_latent + self.task_condition_scale * task_embedding
            detail_latent = detail_latent + 0.5 * self.task_condition_scale * task_embedding
        combined_latent = coarse_latent + torch.tanh(self.refine_gain) * detail_latent
        semantic_input = torch.cat([coarse_latent, detail_latent, slot_gates, torch.stack([complexity_level, complexity_stats['complexity']]), task_one_hot], dim=-1)
        semantic_source = scaled_gradient(self.semantic_input_norm(semantic_input), self.semantic_grad_scale)
        semantic_projection = self.semantic_projector(semantic_source.unsqueeze(0)).squeeze(0)
        semantic_projection = semantic_projection / semantic_projection.norm(p=2).clamp_min(1e-06)
        semantic_query = self.semantic_predictor(semantic_projection.unsqueeze(0)).squeeze(0)
        semantic_query = semantic_query / semantic_query.norm(p=2).clamp_min(1e-06)
        return {'target': target, 'coarse_mu': coarse_mu, 'coarse_logvar': coarse_logvar, 'slot_mu': slot_mu, 'slot_logvar': slot_logvar, 'count_hint': count_hint, 'complexity_hint': complexity_hint, 'complexity_level': complexity_level, 'raw_complexity_level': raw_complexity_level, 'complexity_level_floor': complexity_level_floor, 'complexity_target': complexity_stats['complexity'], 'size_fraction': complexity_stats['size_fraction'], 'spread_complexity': complexity_stats['spread'], 'crowding_complexity': complexity_stats['crowding'], 'task_id': task_id, 'task_one_hot': task_one_hot, 'task_embedding': task_embedding, 'size_prior': size_prior, 'slot_floor': slot_floor, 'slot_gates': slot_gates, 'slot_budget': slot_budget, 'z_slots': z_slots, 'mixture_weights': mixture_weights, 'mixture_entropy': mixture_entropy, 'coarse_latent': coarse_latent, 'detail_latent': detail_latent, 'combined_latent': combined_latent, 'semantic_embedding': semantic_projection, 'semantic_query': semantic_query}

    def _decode(self, state: dict[str, torch.Tensor], decode_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_base = self.backbone.decode(state['coarse_latent'], decode_count)
        if decode_count <= 0:
            empty = state['coarse_latent'].new_zeros((0, self.backbone.input_dim))
            return (coarse_base, coarse_base, empty, state['coarse_latent'].new_tensor(0.0))
        coarse_context = torch.cat([coarse_base, state['coarse_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
        coarse_reconstruction = coarse_base + torch.tanh(self.coarse_refine_gain) * self.coarse_refiner(coarse_context)
        detail_phase = 1.0 if not self.training else self._detail_phase_weight()
        key_indices = self.backbone._key_indices(decode_count, state['coarse_latent'].device)
        queries = self.query_embedding(key_indices)
        attention_scores = queries @ state['z_slots'].transpose(0, 1) / sqrt(self.slot_dim)
        attention_scores = attention_scores + torch.log(state['slot_gates'].clamp_min(0.0001)).unsqueeze(0)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        detail_context = attention_weights @ state['z_slots']
        object_gate = (attention_weights * state['slot_gates'].unsqueeze(0)).sum(dim=-1, keepdim=True)
        residual_context = torch.cat([coarse_reconstruction, detail_context, state['coarse_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
        residual = self.residual_decoder(residual_context)
        reconstruction = coarse_reconstruction + detail_phase * object_gate * torch.tanh(self.residual_gain) * residual
        return (reconstruction, coarse_reconstruction, residual, object_gate.mean() * float(detail_phase))

    def forward(self, x: torch.Tensor, decode_count: int | None=None, decode_mode: str='predicted') -> AutoencoderOutput:
        """Encode, capacity-adapt, decode, and expose auxiliary training losses."""
        if self.training:
            self.train_step += 1
        state = self._encode_state(x, sample=True)
        target = state['target']
        count_features = torch.cat([state['combined_latent'], state['slot_gates']], dim=0)
        shared_count = torch.sigmoid(self.count_regressor(count_features.unsqueeze(0)).squeeze(0)).squeeze(-1)
        if self.count_head_mode == 'shared':
            predicted_count = shared_count
        elif self.count_head_mode == 'detached_aux':
            detached_count_features = torch.cat([state['combined_latent'].detach(), state['slot_gates'].detach(), state['count_hint'].detach().view(1), state['complexity_hint'].detach().view(1)], dim=0)
            predicted_count = torch.sigmoid(self.count_calibrator(detached_count_features.unsqueeze(0)).squeeze(0)).squeeze(-1)
        else:
            raise ValueError(f'Unsupported count_head_mode: {self.count_head_mode}')
        predicted_count = predicted_count * float(self.max_n)
        if decode_count is None:
            if decode_mode == 'predicted':
                decode_count = int(max(0, min(self.max_n, round(float(predicted_count.item())))))
            elif decode_mode == 'target':
                decode_count = int(target.shape[0])
            else:
                raise ValueError(f'Unsupported decode_mode: {decode_mode}')
        reconstruction, coarse_reconstruction, residual, mean_object_gate = self._decode(state, int(decode_count))
        kl_coarse = 0.5 * torch.mean(state['coarse_mu'].pow(2) + state['coarse_logvar'].exp() - 1.0 - state['coarse_logvar'])
        kl_slots = 0.5 * torch.mean(state['slot_mu'].pow(2) + state['slot_logvar'].exp() - 1.0 - state['slot_logvar'])
        coarse_loss = mse_between_sets(target, coarse_reconstruction, 'hungarian')
        final_loss = mse_between_sets(target, reconstruction, 'hungarian')
        kernel_prior = self._heuristic_kernel_weights(state['size_fraction'], state['spread_complexity'], state['crowding_complexity'])
        kernel_weights = self._kernel_weights(state['size_fraction'], state['spread_complexity'], state['crowding_complexity'])
        kernel_prior_error = (kernel_weights - kernel_prior).pow(2).mean() if self.kernel_weight_mode == 'learned_adaptive' else reconstruction.new_tensor(0.0)
        dist_loss = set_mmd(target, reconstruction, weights=kernel_weights)
        moment_loss = moment_alignment(target, reconstruction)
        anchor_loss = self._conditional_anchor_mmd(target, reconstruction, kernel_weights)
        coarse_difficulty = float(coarse_loss.detach().item()) / (float(coarse_loss.detach().item()) + 0.75)
        raw_budget_target = reconstruction.new_tensor(max(0.08, min(0.98, 0.06 + 0.24 * float(state['size_fraction'].item()) + 0.22 * float(state['complexity_target'].item()) + 0.18 * coarse_difficulty)))
        budget_floor = self.budget_floor_alpha * state['size_fraction']
        budget_target = torch.maximum(raw_budget_target, budget_floor)
        budget_loss = (state['slot_budget'] - budget_target).pow(2)
        budget_lower_bound = self.budget_lower_bound_alpha * state['size_fraction'].pow(self.budget_lower_bound_power)
        budget_lower_bound_loss = torch.relu(budget_lower_bound - state['slot_budget']).pow(2)
        improvement_margin = torch.relu(final_loss - coarse_loss + 0.01)
        slot_probs = state['slot_gates'] / state['slot_gates'].sum().clamp_min(1e-06)
        slot_entropy = -torch.sum(slot_probs * torch.log(slot_probs.clamp_min(1e-08))) / torch.log(reconstruction.new_tensor(float(self.num_slots)))
        entropy_target = budget_target.clamp(0.1, 0.9)
        entropy_loss = (slot_entropy - entropy_target).pow(2)
        soft_active_slots = torch.sigmoid(10.0 * (state['slot_gates'] - 0.22)).sum()
        active_target = 1.0 + (self.num_slots - 1) * state['complexity_target']
        active_slot_loss = (soft_active_slots - active_target).pow(2)
        ordering_loss = torch.relu(state['slot_gates'][1:] - state['slot_gates'][:-1]).mean()
        normalized_slots = state['z_slots'] / state['z_slots'].norm(dim=-1, keepdim=True).clamp_min(1e-06)
        slot_similarity = normalized_slots @ normalized_slots.transpose(0, 1)
        diversity_loss = (slot_similarity - torch.eye(self.num_slots, device=target.device, dtype=slot_similarity.dtype)).pow(2).sum() / float(self.num_slots * max(1, self.num_slots - 1))
        complexity_loss = (state['complexity_hint'] - state['complexity_target']).pow(2)
        augmented_state = self._encode_state(self._augment_view(target), sample=False)
        subset_view, removed_view = self._subset_split(target)
        subset_state = self._encode_state(subset_view, sample=False)
        removed_state = self._encode_state(removed_view, sample=False)
        negative_state = self._encode_state(self._negative_view(target), sample=False)
        semantic_cosine = torch.sum(state['semantic_embedding'] * augmented_state['semantic_embedding'])
        subset_cosine = torch.sum(state['semantic_embedding'] * subset_state['semantic_embedding'])
        negative_cosine = torch.sum(state['semantic_embedding'] * negative_state['semantic_embedding'])
        queue_negatives = self._get_semantic_queue().to(target.device)
        shared_negatives = torch.cat([negative_state['semantic_embedding'].detach().unsqueeze(0), queue_negatives], dim=0)
        semantic_targets = torch.stack([state['size_fraction'], state['complexity_target'], state['crowding_complexity']])
        semantic_aux_pred = torch.sigmoid(self.semantic_aux_head(state['semantic_embedding'].unsqueeze(0)).squeeze(0))
        phase_weight = reconstruction.new_tensor(self._semantic_phase_weight())
        semantic_loss = reconstruction.new_tensor(0.0)
        subset_loss = reconstruction.new_tensor(0.0)
        negative_loss = reconstruction.new_tensor(0.0)
        semantic_aux_loss = reconstruction.new_tensor(0.0)
        semantic_margin_loss = reconstruction.new_tensor(0.0)
        dynamic_similarity_gate = reconstruction.new_tensor(1.0)
        dynamic_gate_target = reconstruction.new_tensor(1.0)
        dynamic_gate_reg = reconstruction.new_tensor(0.0)
        if self.semantic_mode != 'off':
            gate_features = torch.stack([state['size_fraction'], state['complexity_target'], state['crowding_complexity'], (1.0 - semantic_cosine.detach()).clamp(0.0, 1.0), (1.0 - subset_cosine.detach()).clamp(0.0, 1.0), negative_cosine.detach().clamp(0.0, 1.0)])
            if self.dynamic_similarity_mode == 'simple':
                gate_features[3:] = 0.0
            dynamic_similarity_gate = self.similarity_gate(gate_features.unsqueeze(0)).squeeze(0).squeeze(0)
            dynamic_similarity_gate = self.dynamic_gate_floor + (self.dynamic_gate_ceiling - self.dynamic_gate_floor) * dynamic_similarity_gate
            dynamic_gate_target = self.dynamic_gate_floor + (self.dynamic_gate_ceiling - self.dynamic_gate_floor) * (0.45 * state['complexity_target'].detach() + 0.35 * negative_cosine.detach().clamp(0.0, 1.0) + 0.2 * (1.0 - subset_cosine.detach()).clamp(0.0, 1.0)).clamp(0.0, 1.0)
            dynamic_gate_reg = (dynamic_similarity_gate - dynamic_gate_target).pow(2)
            dynamic_semantic_weight = phase_weight * (1.0 - self.dynamic_similarity_scale + self.dynamic_similarity_scale * dynamic_similarity_gate)
            base_semantic_weight = phase_weight
        if self.training and self.semantic_mode != 'off' and (float(phase_weight.item()) > 0.0):
            if self.semantic_mode == 'contrastive':
                infonce_aug = info_nce_loss(state['semantic_query'], augmented_state['semantic_embedding'].detach(), shared_negatives, self.semantic_temperature)
                infonce_subset = info_nce_loss(state['semantic_query'], subset_state['semantic_embedding'].detach(), shared_negatives, self.semantic_temperature)
                semantic_loss = base_semantic_weight * (0.65 * infonce_aug + 0.35 * infonce_subset)
                subset_loss = base_semantic_weight * torch.relu(0.94 - subset_cosine)
                negative_loss = dynamic_semantic_weight * torch.relu(negative_cosine - 0.82)
                semantic_margin_loss = dynamic_semantic_weight * torch.relu(0.08 - (semantic_cosine - negative_cosine))
            elif self.semantic_mode == 'cosine':
                semantic_loss = base_semantic_weight * torch.relu(0.992 - semantic_cosine)
                subset_loss = base_semantic_weight * torch.relu(0.985 - subset_cosine)
                negative_loss = dynamic_semantic_weight * torch.relu(negative_cosine - 0.9)
                semantic_margin_loss = dynamic_semantic_weight * torch.relu(0.1 - (semantic_cosine - negative_cosine))
            else:
                raise ValueError(f'Unsupported semantic_mode: {self.semantic_mode}')
            semantic_aux_loss = base_semantic_weight * (semantic_aux_pred - semantic_targets).pow(2).mean()
        monotonicity_loss = torch.relu(subset_state['slot_budget'] - state['slot_budget'] + 0.02)
        hysteresis_latent = (state['combined_latent'] - subset_state['combined_latent'] - removed_state['combined_latent']).pow(2).mean()
        hysteresis_budget = (state['slot_budget'] - subset_state['slot_budget'] - removed_state['slot_budget']).pow(2)
        residual_norm = residual.pow(2).mean() if residual.shape[0] > 0 else reconstruction.new_tensor(0.0)
        detail_phase = reconstruction.new_tensor(self._detail_phase_weight())
        coarse_supervision_weight = reconstruction.new_tensor(self._coarse_supervision_weight())
        step_progress = min(1.0, float(self.train_step.item()) / 80.0)
        dist_weight = reconstruction.new_tensor((0.04 + 0.16 * step_progress) * (0.35 + 0.65 * float(state['complexity_target'].item())))
        if self.distribution_ramp_denom > 0.0:
            distribution_scale = reconstruction.new_tensor(min(1.0, float(target.shape[0]) / float(self.distribution_ramp_denom)))
        else:
            distribution_scale = reconstruction.new_tensor(1.0)
        structure_phase = reconstruction.new_tensor(0.3 + 0.7 * float(detail_phase.item()))
        extra_loss = detail_phase * self.dist_scale * distribution_scale * dist_weight * dist_loss + detail_phase * distribution_scale * self.moment_scale * moment_loss + detail_phase * distribution_scale * self.anchor_scale * anchor_loss + structure_phase * self.budget_scale * budget_loss + structure_phase * self.budget_lower_bound_scale * budget_lower_bound_loss + structure_phase * self.entropy_scale * entropy_loss + structure_phase * self.active_scale * active_slot_loss + structure_phase * self.ordering_scale * ordering_loss + structure_phase * self.diversity_scale * diversity_loss + structure_phase * self.complexity_scale * complexity_loss + detail_phase * self.semantic_scale * semantic_loss + detail_phase * self.subset_scale * subset_loss + detail_phase * self.negative_scale * negative_loss + detail_phase * self.margin_scale * semantic_margin_loss + detail_phase * self.aux_scale * semantic_aux_loss + structure_phase * self.monotonicity_scale * monotonicity_loss + detail_phase * self.hysteresis_scale * hysteresis_latent + detail_phase * self.hysteresis_budget_scale * hysteresis_budget + detail_phase * self.dynamic_gate_reg_scale * dynamic_gate_reg + detail_phase * self.kernel_prior_scale * kernel_prior_error + detail_phase * self.improvement_scale * improvement_margin + detail_phase * self.residual_scale * residual_norm + coarse_supervision_weight * self.coarse_supervision_scale * coarse_loss
        if self.training:
            self._enqueue_semantic(state['semantic_embedding'].detach())
        self.last_aux = {'kl': kl_coarse + structure_phase * kl_slots, 'extra_loss': extra_loss, 'stats': {'slot_budget': float(state['slot_budget'].item()), 'budget_floor': float(budget_floor.item()), 'raw_budget_target': float(raw_budget_target.item()), 'budget_target': float(budget_target.item()), 'budget_lower_bound': float(budget_lower_bound.item()), 'budget_lower_bound_loss': float(budget_lower_bound_loss.item()), 'count_hint': float(state['count_hint'].item()), 'shared_count': float(shared_count.item() * float(self.max_n)), 'complexity_hint': float(state['complexity_hint'].item()), 'complexity_target': float(state['complexity_target'].item()), 'complexity_level': float(state['complexity_level'].item()), 'raw_complexity_level': float(state['raw_complexity_level'].item()), 'complexity_level_floor': float(state['complexity_level_floor'].item()), 'spread_complexity': float(state['spread_complexity'].item()), 'crowding_complexity': float(state['crowding_complexity'].item()), 'size_prior': float(state['size_prior'].item()), 'slot_floor_mean': float(state['slot_floor'].mean().item()), 'slot_entropy': float(slot_entropy.item()), 'active_slots@0.25': float((state['slot_gates'] > 0.25).float().sum().item()), 'soft_active_slots': float(soft_active_slots.item()), 'active_target': float(active_target.item()), 'semantic_cosine': float(semantic_cosine.item()), 'subset_cosine': float(subset_cosine.item()), 'negative_cosine': float(negative_cosine.item()), 'detail_phase_weight': float(detail_phase.item()), 'coarse_supervision_weight': float(coarse_supervision_weight.item()), 'structure_phase_weight': float(structure_phase.item()), 'semantic_phase_weight': float(phase_weight.item()), 'dynamic_similarity_gate': float(dynamic_similarity_gate.item()), 'dynamic_gate_target': float(dynamic_gate_target.item()), 'dynamic_gate_reg': float(dynamic_gate_reg.item()), 'semantic_temperature': float(self.semantic_temperature), 'semantic_queue_count': float(self.semantic_queue_count.item()), 'subset_budget': float(subset_state['slot_budget'].item()), 'dist_mmd': float(dist_loss.item()), 'distribution_scale': float(distribution_scale.item()), 'moment_alignment': float(moment_loss.item()), 'anchor_mmd': float(anchor_loss.item()), 'task_id': float(state['task_id'].item()), 'hysteresis_latent': float(hysteresis_latent.item()), 'hysteresis_budget': float(hysteresis_budget.item()), 'kernel_weight_local': float(kernel_weights[0].item()), 'kernel_weight_medium': float(kernel_weights[1].item()), 'kernel_weight_broad': float(kernel_weights[2].item()), 'kernel_weight_global': float(kernel_weights[3].item()), 'kernel_prior_error': float(kernel_prior_error.item()), 'mixture_entropy': float(state['mixture_entropy'].item()), 'mixture_weight_peak': float(state['mixture_weights'].max().item()), 'coarse_loss': float(coarse_loss.item()), 'coarse_refine_gain': float(torch.tanh(self.coarse_refine_gain).item()), 'residual_norm': float(residual_norm.item()), 'diversity_loss': float(diversity_loss.item()), 'refine_gain': float(torch.tanh(self.refine_gain).item()), 'residual_gain': float(torch.tanh(self.residual_gain).item()), 'mean_object_gate': float(mean_object_gate.item())}}
        self.last_aux_target = target.detach()
        self.last_aux_reconstruction = reconstruction.detach()
        return AutoencoderOutput(reconstruction=reconstruction, latent=state['combined_latent'], predicted_count=predicted_count, target=target, decode_count=int(decode_count))


class TACDBackbone(BranchSetAutoencoder):
    """Hierarchical set autoencoder backbone with task and distribution losses."""

    def __init__(self, input_dim: int=6, hidden_dim: int=128, latent_dim: int=96, num_slots: int=4, slot_dim: int=16, semantic_dim: int=64, max_n: int=16, total_steps_hint: int=4500, coarse_dim_ratio: float=8.0 / 96.0, coarse_refine_layers: int=2, use_hvq: bool=True, hvq_num_codes: int=64, hvq_commitment_scale: float=0.25, hvq_codebook_scale: float=1.0, hvq_loss_scale: float=0.08, hvq_entropy_scale: float=0.01, hvq_bandwidth_threshold: float=0.35, hvq_fine_residual_scale: float=0.65, hvq_outlier_injection_scale: float=0.5, hvq_query_mode: str='hybrid', hvq_query_embed_blend: float=0.0, hvq_query_train_max: int=12, hvq_residual_reg_scale: float=0.02, hvq_variable_capacity: bool=False, hvq_min_active_slots: int=1, hvq_capacity_signal_bias: float=0.0, hvq_capacity_signal_scale: float=1.0, hvq_outlier_score_bonus: float=0.15, hvq_variable_capacity_loss_scale: float=0.0, hvq_variable_capacity_temperature: float=0.12, hvq_variable_capacity_threshold: float=0.5, phase_a_end: float=0.35, phase_b_end: float=0.7, max_blend: float=0.5, detail_loss_scale: float=0.4, budget_floor_alpha: float=0.55, budget_floor_power: float=0.65, coarse_sup_peak: float=0.3, coarse_sup_floor: float=0.08, coarse_sup_decay_end: float=0.85, semantic_start: float=0.35, semantic_ramp_duration: float=0.2, semantic_mode: str='contrastive', semantic_positive_style: str='subset_noise', semantic_temperature: float=0.15, semantic_grad_scale: float=0.1, semantic_queue_size: int=128, neg_cosine_threshold: float=0.5, margin_gap: float=0.2, repulsion_scale: float=0.12, prenorm_var_scale: float=0.1, prenorm_var_target: float=1.0, dist_scale: float=0.85, moment_scale: float=0.14, anchor_scale: float=0.14, kernel_prior_scale: float=0.05, budget_scale: float=0.14, entropy_scale: float=0.08, active_scale: float=0.14, complexity_scale: float=0.1, semantic_scale: float=0.3, subset_scale: float=0.15, negative_scale: float=0.2, margin_scale: float=0.15, hysteresis_scale: float=0.16, hysteresis_budget_scale: float=0.1, hysteresis_recon_scale: float=0.02, hysteresis_slot_scale: float=0.0, hysteresis_gate_scale: float=0.0, hysteresis_refresh_blend: float=0.65, monotonicity_scale: float=0.08, improvement_scale: float=0.14, distribution_ramp_denom: float=6.0, dynamic_similarity_scale: float=0.65, dynamic_similarity_mode: str='pair_adaptive', dynamic_gate_floor: float=0.18, dynamic_gate_ceiling: float=0.82, dynamic_gate_reg_scale: float=0.06, kernel_weight_mode: str='learned_adaptive', kernel_prior_mix: float=0.82, anchor_mode: str='sign_binary', task_condition_mode: str='synthetic_stats', task_condition_scale: float=0.1, slot_aggregation_mode: str='mixture_softmax', mixture_temperature: float=1.15):
        actual_coarse_dim = max(8, min(latent_dim, int(latent_dim * coarse_dim_ratio)))
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_slots=num_slots, slot_dim=slot_dim, semantic_dim=semantic_dim, max_n=max_n, total_steps_hint=total_steps_hint, semantic_mode=semantic_mode, semantic_positive_style=semantic_positive_style, semantic_temperature=semantic_temperature, semantic_grad_scale=semantic_grad_scale, semantic_queue_size=semantic_queue_size, semantic_start_frac=0.0, semantic_warmup_frac=0.0, task_condition_mode=task_condition_mode, task_condition_scale=task_condition_scale, coarse_warmup_frac=0.0, coarse_supervision_scale=0.0, coarse_supervision_frac=0.0, coarse_supervision_floor=0.0, complexity_level_floor_alpha=0.0, complexity_level_floor_power=1.0, dynamic_similarity_scale=dynamic_similarity_scale, dynamic_similarity_mode=dynamic_similarity_mode, dynamic_gate_floor=dynamic_gate_floor, dynamic_gate_ceiling=dynamic_gate_ceiling, dynamic_gate_reg_scale=dynamic_gate_reg_scale, count_head_mode='shared', kernel_weight_mode=kernel_weight_mode, kernel_prior_mix=kernel_prior_mix, kernel_prior_scale=kernel_prior_scale, anchor_mode=anchor_mode, anchor_scale=anchor_scale, distribution_ramp_denom=distribution_ramp_denom, hysteresis_scale=hysteresis_scale, hysteresis_budget_scale=hysteresis_budget_scale, dist_scale=dist_scale, moment_scale=moment_scale, budget_scale=budget_scale, budget_floor_alpha=0.0, budget_lower_bound_alpha=0.0, budget_lower_bound_power=1.0, budget_lower_bound_scale=0.0, slot_gate_floor_alpha=0.0, slot_gate_floor_scale=0.0, entropy_scale=entropy_scale, active_scale=active_scale, ordering_scale=0.0, diversity_scale=0.0, complexity_scale=complexity_scale, semantic_scale=semantic_scale, subset_scale=subset_scale, negative_scale=negative_scale, margin_scale=margin_scale, aux_scale=0.0, monotonicity_scale=monotonicity_scale, improvement_scale=improvement_scale, residual_scale=0.0, slot_aggregation_mode=slot_aggregation_mode, mixture_temperature=mixture_temperature)
        self.coarse_dim = actual_coarse_dim
        self.coarse_mu = _base_mlp(latent_dim, hidden_dim, self.coarse_dim)
        self.coarse_logvar = _base_mlp(latent_dim, hidden_dim, self.coarse_dim)
        self.coarse_to_latent = _base_mlp(self.coarse_dim, hidden_dim, latent_dim)
        self.use_hvq = bool(use_hvq)
        self.hvq_num_codes = int(hvq_num_codes)
        self.hvq_commitment_scale = float(hvq_commitment_scale)
        self.hvq_codebook_scale = float(hvq_codebook_scale)
        self.hvq_loss_scale = float(hvq_loss_scale)
        self.hvq_entropy_scale = float(hvq_entropy_scale)
        self.hvq_bandwidth_threshold = float(hvq_bandwidth_threshold)
        self.hvq_fine_residual_scale = float(hvq_fine_residual_scale)
        self.hvq_outlier_injection_scale = float(hvq_outlier_injection_scale)
        self.hvq_query_mode = str(hvq_query_mode)
        self.hvq_query_embed_blend = float(hvq_query_embed_blend)
        self.hvq_query_train_max = int(hvq_query_train_max)
        self.hvq_residual_reg_scale = float(hvq_residual_reg_scale)
        self.hvq_variable_capacity = bool(hvq_variable_capacity)
        self.hvq_min_active_slots = max(1, int(hvq_min_active_slots))
        self.hvq_capacity_signal_bias = float(hvq_capacity_signal_bias)
        self.hvq_capacity_signal_scale = float(hvq_capacity_signal_scale)
        self.hvq_outlier_score_bonus = float(hvq_outlier_score_bonus)
        self.hvq_variable_capacity_loss_scale = float(hvq_variable_capacity_loss_scale)
        self.hvq_variable_capacity_temperature = float(hvq_variable_capacity_temperature)
        self.hvq_variable_capacity_threshold = float(hvq_variable_capacity_threshold)
        self.hvq_codebook = torch.nn.Embedding(self.hvq_num_codes, slot_dim)
        torch.nn.init.uniform_(self.hvq_codebook.weight, -1.0 / max(1, self.hvq_num_codes), 1.0 / max(1, self.hvq_num_codes))
        self.hvq_outlier_encoder = _base_mlp(input_dim + 2, hidden_dim, slot_dim)
        self.hvq_rank_query = _base_mlp(4, hidden_dim, slot_dim)
        self.coarse_refine_layers = coarse_refine_layers
        self.coarse_refiner = _base_mlp(input_dim + latent_dim, hidden_dim, input_dim)
        if coarse_refine_layers >= 2:
            self.coarse_refiner_2 = _base_mlp(input_dim + latent_dim + input_dim, hidden_dim, input_dim)
            self.coarse_refine_gain_2 = torch.nn.Parameter(torch.tensor(0.15))
        self.detail_refiner = _base_mlp(input_dim + latent_dim, hidden_dim, input_dim)
        self.detail_refine_gain = torch.nn.Parameter(torch.tensor(0.25))
        self.semantic_projector = torch.nn.Sequential(torch.nn.Linear(self.semantic_input_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, hidden_dim // 2), torch.nn.LayerNorm(hidden_dim // 2), torch.nn.GELU(), torch.nn.Linear(hidden_dim // 2, semantic_dim))
        self.semantic_predictor = torch.nn.Sequential(torch.nn.Linear(semantic_dim, hidden_dim), torch.nn.LayerNorm(hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, hidden_dim // 2), torch.nn.LayerNorm(hidden_dim // 2), torch.nn.GELU(), torch.nn.Linear(hidden_dim // 2, semantic_dim))
        self.phase_a_end = phase_a_end
        self.phase_b_end = phase_b_end
        self.max_blend = max_blend
        self.detail_loss_scale = detail_loss_scale
        self.coarse_sup_peak = coarse_sup_peak
        self.coarse_sup_floor = coarse_sup_floor
        self.coarse_sup_decay_end = coarse_sup_decay_end
        self.semantic_start = semantic_start
        self.semantic_ramp_duration = semantic_ramp_duration
        self.neg_cosine_threshold = neg_cosine_threshold
        self.margin_gap = margin_gap
        self.repulsion_scale = repulsion_scale
        self.prenorm_var_scale = prenorm_var_scale
        self.prenorm_var_target = prenorm_var_target
        self.opt_budget_floor_alpha = budget_floor_alpha
        self.opt_budget_floor_power = budget_floor_power
        self.s = type('S', (), {})()
        self.s.dist = dist_scale
        self.s.moment = moment_scale
        self.s.anchor = anchor_scale
        self.s.kernel_prior = kernel_prior_scale
        self.s.budget = budget_scale
        self.s.entropy = entropy_scale
        self.s.active = active_scale
        self.s.complexity = complexity_scale
        self.s.semantic = semantic_scale
        self.s.subset = subset_scale
        self.s.negative = negative_scale
        self.s.margin = margin_scale
        self.s.hysteresis = hysteresis_scale
        self.s.hysteresis_budget = hysteresis_budget_scale
        self.hysteresis_recon_scale = float(hysteresis_recon_scale)
        self.hysteresis_slot_scale = float(hysteresis_slot_scale)
        self.hysteresis_gate_scale = float(hysteresis_gate_scale)
        self.hysteresis_refresh_blend = float(hysteresis_refresh_blend)
        self.s.monotonicity = monotonicity_scale
        self.s.improvement = improvement_scale
        self.s.gate_reg = dynamic_gate_reg_scale

    def _progress(self) -> float:
        return min(1.0, float(self.train_step.item()) / float(self.total_steps_hint))

    def _blend_weight(self) -> float:
        p = self._progress()
        if p < self.phase_a_end:
            return 0.0
        if p < self.phase_b_end:
            t = (p - self.phase_a_end) / (self.phase_b_end - self.phase_a_end)
            return self.max_blend * t
        return self.max_blend

    def _semantic_weight(self) -> float:
        p = self._progress()
        if p < self.semantic_start:
            return 0.0
        t = (p - self.semantic_start) / max(0.01, self.semantic_ramp_duration)
        return min(1.0, t)

    def _coarse_sup_weight(self) -> float:
        p = self._progress()
        if p <= self.phase_a_end:
            return self.coarse_sup_peak
        if p >= self.coarse_sup_decay_end:
            return self.coarse_sup_floor
        t = (p - self.phase_a_end) / (self.coarse_sup_decay_end - self.phase_a_end)
        return self.coarse_sup_peak + (self.coarse_sup_floor - self.coarse_sup_peak) * t

    def _hvq_outlier_slots(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-slot summaries for concrete outlier/high-value elements."""
        if target.shape[0] == 0:
            return (target.new_zeros((self.num_slots, self.slot_dim)), target.new_zeros(self.num_slots), target.new_tensor(0.0))
        centered = target - target.mean(dim=0, keepdim=True)
        radius = centered.pow(2).mean(dim=-1).sqrt()
        if target.shape[0] <= 1:
            nearest = target.new_zeros(target.shape[0])
        else:
            pairwise = torch.cdist(target, target)
            inf = torch.full_like(pairwise, float('inf'))
            nearest = torch.where(torch.eye(target.shape[0], device=target.device, dtype=torch.bool), inf, pairwise).min(dim=1).values
        raw_scores = radius + 0.5 * nearest
        norm_scores = (raw_scores - raw_scores.mean()) / raw_scores.std(unbiased=False).clamp_min(1e-06)
        k = min(self.num_slots, target.shape[0])
        top_scores, top_idx = torch.topk(raw_scores, k=k)
        selected = target.new_zeros((self.num_slots, target.shape[1]))
        selected_scores = target.new_zeros(self.num_slots)
        active_mask = target.new_zeros(self.num_slots)
        selected[:k] = target[top_idx]
        selected_scores[:k] = norm_scores[top_idx]
        active_mask[:k] = 1.0
        features = torch.cat([selected, selected_scores.unsqueeze(-1), self.slot_positions.to(device=target.device, dtype=target.dtype).unsqueeze(-1)], dim=-1)
        outlier_slots = self.hvq_outlier_encoder(features) * active_mask.unsqueeze(-1)
        return (outlier_slots, active_mask, top_scores.mean())

    def _hvq_quantize(self, z_slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize fine-detail slots with a straight-through VQ codebook."""
        if not self.use_hvq or z_slots.numel() == 0:
            zero = z_slots.new_tensor(0.0)
            return {'z_q': z_slots, 'loss': zero, 'commitment': zero, 'codebook': zero, 'usage': zero, 'perplexity': zero, 'indices': torch.zeros(z_slots.shape[0], device=z_slots.device, dtype=torch.long)}
        codebook = self.hvq_codebook.weight.to(device=z_slots.device, dtype=z_slots.dtype)
        distances = z_slots.pow(2).sum(dim=1, keepdim=True) - 2.0 * z_slots @ codebook.t() + codebook.pow(2).sum(dim=1).unsqueeze(0)
        indices = torch.argmin(distances, dim=1)
        z_nearest = self.hvq_codebook(indices).to(dtype=z_slots.dtype)
        z_q = z_slots + (z_nearest - z_slots).detach()
        commitment = (z_slots - z_nearest.detach()).pow(2).mean()
        codebook_loss = (z_nearest - z_slots.detach()).pow(2).mean()
        one_hot = torch.nn.functional.one_hot(indices, num_classes=self.hvq_num_codes).to(z_slots.dtype)
        avg_probs = one_hot.mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs.clamp_min(1e-08))).sum()
        usage = 1.0 - entropy / torch.log(z_slots.new_tensor(float(self.hvq_num_codes)))
        perplexity = torch.exp(entropy)
        loss = self.hvq_codebook_scale * codebook_loss + self.hvq_commitment_scale * commitment
        return {'z_q': z_q, 'loss': loss, 'commitment': commitment, 'codebook': codebook_loss, 'usage': usage, 'perplexity': perplexity, 'indices': indices}

    def _hvq_decode_queries(self, decode_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """OOD-stable queries for HVQ fine-slot decoding.

        Learned per-index queries are brittle when training only sees n<=12
        and OOD evaluation decodes n=100. Hybrid mode keeps the learned queries
        for the trained prefix and uses normalized rank features past that
        prefix, so OOD positions stay inside the same [0, 1] support.
        """
        key_indices = self.backbone._key_indices(decode_count, device)
        if self.hvq_query_mode == 'embedding':
            return self.query_embedding(key_indices).to(dtype=dtype)
        rank = torch.arange(1, decode_count + 1, device=device, dtype=dtype)
        rank = rank / float(max(1, decode_count))
        centered = 2.0 * rank - 1.0
        features = torch.stack([rank, centered, torch.sin(rank * pi), torch.cos(rank * pi)], dim=-1)
        rank_queries = self.hvq_rank_query(features)
        if self.hvq_query_mode == 'hybrid':
            train_max = max(0, min(int(self.hvq_query_train_max), decode_count))
            if train_max > 0:
                rank_queries = rank_queries.clone()
                rank_queries[:train_max] = self.query_embedding(key_indices[:train_max]).to(dtype=dtype)
        blend = max(0.0, min(1.0, self.hvq_query_embed_blend))
        if blend <= 0.0:
            return rank_queries
        embed_queries = self.query_embedding(key_indices).to(dtype=dtype)
        return (1.0 - blend) * rank_queries + blend * embed_queries

    def _hvq_transmit_gates(self, state: dict[str, torch.Tensor], slot_gates: torch.Tensor, outlier_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Select the actually transmitted fine slots.

        When enabled, this converts the soft slot bank into a bounded
        variable-length message: the tensor shape remains fixed for PyTorch, but
        only top-k slots have a nonzero forward-pass gate; active amplitudes
        keep the learned soft gate values.
        """
        zero = slot_gates.new_tensor(0.0)
        if not self.use_hvq or not self.hvq_variable_capacity:
            return (slot_gates, {'enabled': zero, 'target_slots': slot_gates.sum(), 'active_slots': (slot_gates > 0.25).to(slot_gates.dtype).sum(), 'rate_loss': zero, 'soft_budget': slot_gates.mean(), 'hard_mask': (slot_gates > 0.25).to(slot_gates.dtype)})
        min_slots = max(1, min(self.num_slots, self.hvq_min_active_slots))
        signal = 0.2 * state['size_fraction'] + 0.35 * state['complexity_target'] + 0.45 * state['complexity_level']
        signal = (self.hvq_capacity_signal_bias + self.hvq_capacity_signal_scale * signal).clamp(0.0, 1.0)
        target_slots = (slot_gates.new_tensor(float(min_slots)) + slot_gates.new_tensor(float(self.num_slots - min_slots)) * signal).clamp(float(min_slots), float(self.num_slots))
        k = int(torch.ceil(target_slots.detach()).clamp(float(min_slots), float(self.num_slots)).item())
        scores = slot_gates + self.hvq_outlier_score_bonus * outlier_mask.to(slot_gates.dtype)
        _values, indices = torch.topk(scores, k=k)
        hard_mask = torch.zeros_like(slot_gates)
        hard_mask.scatter_(0, indices, 1.0)
        transmit_gates = hard_mask * slot_gates
        temp = max(0.0001, self.hvq_variable_capacity_temperature)
        soft_active = torch.sigmoid((scores - self.hvq_variable_capacity_threshold) / temp).sum()
        rate_loss = (soft_active - target_slots.detach()).pow(2)
        return (transmit_gates, {'enabled': slot_gates.new_tensor(1.0), 'target_slots': target_slots, 'active_slots': hard_mask.sum(), 'rate_loss': rate_loss, 'soft_budget': slot_gates.mean(), 'hard_mask': hard_mask, 'capacity_signal': signal, 'soft_active_slots': soft_active})

    def _encode_state(self, x: torch.Tensor, sample: bool) -> dict[str, torch.Tensor]:
        state = super()._encode_state(x, sample)
        size_frac = state['size_fraction']
        budget_floor = self.opt_budget_floor_alpha * size_frac.pow(self.opt_budget_floor_power)
        gate_floor = budget_floor * torch.sigmoid(8.0 * (size_frac - 0.15 * self.slot_positions))
        floored_gates = torch.maximum(state['slot_gates'], gate_floor)
        state['slot_gates_soft'] = floored_gates
        state['budget_floor_applied'] = budget_floor
        outlier_slots, outlier_mask, outlier_strength = self._hvq_outlier_slots(state['target'])
        transmit_gates, capacity_state = self._hvq_transmit_gates(state, floored_gates, outlier_mask)
        state['slot_gates'] = transmit_gates
        state['slot_budget'] = transmit_gates.mean()
        state['hvq_transmit_mask'] = capacity_state['hard_mask']
        state['hvq_variable_capacity_enabled'] = capacity_state['enabled']
        state['hvq_target_active_slots'] = capacity_state['target_slots']
        state['hvq_active_slots'] = capacity_state['active_slots']
        state['hvq_capacity_rate_loss'] = capacity_state['rate_loss']
        state['hvq_soft_slot_budget'] = capacity_state['soft_budget']
        state['hvq_capacity_signal'] = capacity_state.get('capacity_signal', state['slot_budget'])
        state['hvq_soft_transmit_slots'] = capacity_state.get('soft_active_slots', capacity_state['active_slots'])
        fine_slots = state['z_slots']
        if self.use_hvq:
            fine_slots = fine_slots + self.hvq_outlier_injection_scale * transmit_gates.unsqueeze(-1) * outlier_slots
        hvq = self._hvq_quantize(fine_slots)
        state['z_slots_raw'] = state['z_slots']
        state['hvq_outlier_slots'] = outlier_slots
        state['hvq_outlier_mask'] = outlier_mask
        state['hvq_outlier_strength'] = outlier_strength
        state['hvq_fine_slots_prequant'] = fine_slots
        state['z_slots'] = hvq['z_q'] * transmit_gates.unsqueeze(-1) if self.use_hvq else hvq['z_q']
        state['hvq_loss'] = hvq['loss']
        state['hvq_commitment_loss'] = hvq['commitment']
        state['hvq_codebook_loss'] = hvq['codebook']
        state['hvq_usage_loss'] = hvq['usage']
        state['hvq_perplexity'] = hvq['perplexity']
        state['hvq_code_indices'] = hvq['indices']
        state['hvq_active_codes'] = hvq['indices'].unique().numel() if hvq['indices'].numel() else 0
        detail_summary, mixture_weights, mixture_entropy = self._aggregate_slots(state['z_slots'], transmit_gates, state['complexity_level'].unsqueeze(0), size_frac.unsqueeze(0))
        state['mixture_weights'] = mixture_weights
        state['mixture_entropy'] = mixture_entropy
        detail_latent = self.slot_summary_to_latent(detail_summary)
        coarse_latent = state['coarse_latent']
        if self.task_condition_mode != 'off':
            detail_latent = detail_latent + 0.5 * self.task_condition_scale * state['task_embedding']
        state['detail_latent'] = detail_latent
        state['combined_latent'] = coarse_latent + torch.tanh(self.refine_gain) * detail_latent
        state['hvq_bandwidth_gate'] = torch.sigmoid(12.0 * (state['slot_budget'] - state['slot_budget'].new_tensor(self.hvq_bandwidth_threshold))) if self.use_hvq else state['slot_budget'].new_tensor(0.0)
        semantic_input = torch.cat([state['coarse_latent'], state['detail_latent'], state['slot_gates'], torch.stack([state['complexity_level'], state['complexity_target']]), state['task_one_hot']], dim=-1)
        semantic_source = scaled_gradient(self.semantic_input_norm(semantic_input), self.semantic_grad_scale)
        raw_projection = self.semantic_projector(semantic_source.unsqueeze(0)).squeeze(0)
        semantic_projection = raw_projection / raw_projection.norm(p=2).clamp_min(1e-06)
        semantic_query = self.semantic_predictor(semantic_projection.unsqueeze(0)).squeeze(0)
        semantic_query = semantic_query / semantic_query.norm(p=2).clamp_min(1e-06)
        state['raw_semantic_projection'] = raw_projection
        state['semantic_embedding'] = semantic_projection
        state['semantic_query'] = semantic_query
        return state

    def _decode_tacd_autoencoder(self, state: dict[str, torch.Tensor], decode_count: int) -> dict[str, torch.Tensor]:
        if decode_count <= 0:
            empty = state['coarse_latent'].new_zeros((0, self.backbone.input_dim))
            zero = state['coarse_latent'].new_tensor(0.0)
            return {'reconstruction': empty, 'coarse_recon': empty, 'detail_recon': empty, 'blend_weight': zero}
        coarse_base = self.backbone.decode(state['coarse_latent'], decode_count)
        coarse_ctx = torch.cat([coarse_base, state['coarse_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
        refined = coarse_base + torch.tanh(self.coarse_refine_gain) * self.coarse_refiner(coarse_ctx)
        if self.coarse_refine_layers >= 2:
            ctx2 = torch.cat([refined, state['coarse_latent'].unsqueeze(0).expand(decode_count, -1), coarse_base], dim=-1)
            coarse_recon = refined + torch.tanh(self.coarse_refine_gain_2) * self.coarse_refiner_2(ctx2)
        else:
            coarse_recon = refined
        detail_base = self.backbone.decode(state['combined_latent'], decode_count)
        detail_ctx = torch.cat([detail_base, state['combined_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
        detail_recon = detail_base + torch.tanh(self.detail_refine_gain) * self.detail_refiner(detail_ctx)
        fine_residual_norm = state['coarse_latent'].new_tensor(0.0)
        fine_gate_mean = state['coarse_latent'].new_tensor(0.0)
        if self.use_hvq:
            queries = self._hvq_decode_queries(decode_count, state['coarse_latent'].device, state['coarse_latent'].dtype)
            attn_scores = queries @ state['z_slots'].transpose(0, 1) / float(self.slot_dim) ** 0.5
            attn_scores = attn_scores + torch.log(state['slot_gates'].clamp_min(0.0001)).unsqueeze(0)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            fine_context = attn_weights @ state['z_slots']
            object_gate = (attn_weights * state['slot_gates'].unsqueeze(0)).sum(dim=-1, keepdim=True)
            fine_ctx = torch.cat([detail_recon, fine_context, state['combined_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
            fine_residual = torch.tanh(self.residual_decoder(fine_ctx))
            fine_gate = state['hvq_bandwidth_gate'] * object_gate
            detail_recon = detail_recon + self.hvq_fine_residual_scale * fine_gate * torch.tanh(self.residual_gain) * fine_residual
            fine_residual_norm = fine_residual.pow(2).mean()
            fine_gate_mean = fine_gate.mean()
        blend = self._blend_weight()
        blend_t = state['coarse_latent'].new_tensor(blend)
        reconstruction = (1.0 - blend_t) * coarse_recon + blend_t * detail_recon
        return {'reconstruction': reconstruction, 'coarse_recon': coarse_recon, 'detail_recon': detail_recon, 'blend_weight': blend_t, 'fine_residual_norm': fine_residual_norm, 'fine_gate_mean': fine_gate_mean}

    def _hysteresis_remainder_state(self, full_state: dict[str, torch.Tensor], subset_state: dict[str, torch.Tensor], removed_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """State used to train sudden-removal behavior.

        It simulates the cached online operation "encode the crowded set, then
        subtract the removed agents" and asks that state to decode like a clean
        direct re-encoding of the remaining subset.
        """
        remainder = dict(subset_state)
        full_count = full_state['count_hint'].clamp_min(1.0)
        drop_ratio = (removed_state['count_hint'] / full_count).clamp(0.0, 1.0)
        max_refresh = full_count.new_tensor(max(0.0, min(1.0, self.hysteresis_refresh_blend)))
        refresh = torch.minimum(1.25 * drop_ratio, max_refresh)
        residual_coarse = full_state['coarse_latent'] - removed_state['coarse_latent']
        residual_detail = full_state['detail_latent'] - removed_state['detail_latent']
        coarse_latent = (1.0 - refresh) * residual_coarse + refresh * subset_state['coarse_latent']
        detail_latent = (1.0 - refresh) * residual_detail + refresh * subset_state['detail_latent']
        remainder['coarse_latent'] = coarse_latent
        remainder['detail_latent'] = detail_latent
        remainder['combined_latent'] = coarse_latent + torch.tanh(self.refine_gain) * detail_latent
        remainder['hysteresis_refresh'] = refresh
        residual_gates = torch.relu(full_state['slot_gates'] - removed_state['slot_gates'])
        slot_gates = (1.0 - refresh) * residual_gates + refresh * subset_state['slot_gates']
        remainder['slot_gates'] = slot_gates.clamp(max=1.0)
        remainder['slot_budget'] = remainder['slot_gates'].mean()
        if self.use_hvq:
            residual_prequant = full_state.get('hvq_fine_slots_prequant', full_state['z_slots']) - removed_state.get('hvq_fine_slots_prequant', removed_state['z_slots'])
            subset_prequant = subset_state.get('hvq_fine_slots_prequant', subset_state['z_slots'])
            prequant = (1.0 - refresh) * residual_prequant + refresh * subset_prequant
            hvq = self._hvq_quantize(prequant)
            remainder['hvq_fine_slots_prequant'] = prequant
            remainder['z_slots'] = hvq['z_q']
            remainder['hvq_code_indices'] = hvq['indices']
            remainder['hvq_active_codes'] = hvq['indices'].unique().numel() if hvq['indices'].numel() else 0
            remainder['hvq_perplexity'] = hvq['perplexity']
            remainder['hvq_bandwidth_gate'] = torch.sigmoid(12.0 * (remainder['slot_budget'] - remainder['slot_budget'].new_tensor(self.hvq_bandwidth_threshold)))
        return remainder

    def forward(self, x: torch.Tensor, decode_count: int | None=None, decode_mode: str='predicted') -> 'AutoencoderOutput':
        """Run TACD reconstruction and publish losses used by experiment runners."""
        if should_advance_schedule(self.training) and self.train_step < self.total_steps_hint:
            self.train_step += 1
        state = self._encode_state(x, sample=True)
        target = state['target']
        count_features = torch.cat([state['combined_latent'], state['slot_gates']], dim=0)
        predicted_count = torch.sigmoid(self.count_regressor(count_features.unsqueeze(0)).squeeze(0)).squeeze(-1) * float(self.max_n)
        if decode_count is None:
            if decode_mode == 'predicted':
                decode_count = int(max(0, min(self.max_n, round(float(predicted_count.item())))))
            elif decode_mode == 'target':
                decode_count = int(target.shape[0])
            else:
                raise ValueError(f'Unsupported decode_mode: {decode_mode}')
        dec = self._decode_tacd_autoencoder(state, int(decode_count))
        reconstruction = dec['reconstruction']
        coarse_recon = dec['coarse_recon']
        detail_recon = dec['detail_recon']
        blend_w = dec['blend_weight']
        fine_residual_norm = dec.get('fine_residual_norm', reconstruction.new_tensor(0.0))
        fine_gate_mean = dec.get('fine_gate_mean', reconstruction.new_tensor(0.0))
        kl_coarse = 0.5 * torch.mean(state['coarse_mu'].pow(2) + state['coarse_logvar'].exp() - 1.0 - state['coarse_logvar'])
        kl_slots = 0.5 * torch.mean(state['slot_mu'].pow(2) + state['slot_logvar'].exp() - 1.0 - state['slot_logvar'])
        coarse_loss = mse_between_sets(target, coarse_recon, 'hungarian')
        detail_loss = mse_between_sets(target, detail_recon, 'hungarian')
        final_loss = mse_between_sets(target, reconstruction, 'hungarian')
        detail_delta = coarse_loss - detail_loss
        progress = self._progress()
        sem_w = reconstruction.new_tensor(self._semantic_weight())
        coarse_sup_w = reconstruction.new_tensor(self._coarse_sup_weight())
        coarse_diff = float(coarse_loss.detach().item()) / (float(coarse_loss.detach().item()) + 0.75)
        raw_budget_target = reconstruction.new_tensor(max(0.08, min(0.98, 0.06 + 0.24 * float(state['size_fraction'].item()) + 0.22 * float(state['complexity_target'].item()) + 0.18 * coarse_diff)))
        budget_floor = self.opt_budget_floor_alpha * state['size_fraction'].pow(self.opt_budget_floor_power)
        budget_target = torch.maximum(raw_budget_target, budget_floor)
        budget_loss = (state['slot_budget'] - budget_target).pow(2)
        hvq_loss = state.get('hvq_loss', reconstruction.new_tensor(0.0))
        hvq_usage_loss = state.get('hvq_usage_loss', reconstruction.new_tensor(0.0))
        hvq_capacity_rate_loss = state.get('hvq_capacity_rate_loss', reconstruction.new_tensor(0.0))
        hvq_residual_reg = self.hvq_residual_reg_scale * fine_residual_norm if self.use_hvq else reconstruction.new_tensor(0.0)
        slot_probs = state['slot_gates'] / state['slot_gates'].sum().clamp_min(1e-06)
        slot_entropy = -torch.sum(slot_probs * torch.log(slot_probs.clamp_min(1e-08))) / torch.log(reconstruction.new_tensor(float(self.num_slots)))
        entropy_loss = (slot_entropy - budget_target.clamp(0.1, 0.9)).pow(2)
        default_soft_active_slots = torch.sigmoid(10.0 * (state['slot_gates'] - 0.22)).sum()
        soft_active_slots = state.get('hvq_soft_transmit_slots', default_soft_active_slots) if self.hvq_variable_capacity else default_soft_active_slots
        active_target = 1.0 + (self.num_slots - 1) * state['complexity_target']
        active_slot_loss = (soft_active_slots - active_target).pow(2)
        complexity_loss = (state['complexity_hint'] - state['complexity_target']).pow(2)
        kernel_weights = self._kernel_weights(state['size_fraction'], state['spread_complexity'], state['crowding_complexity'])
        kernel_prior = self._heuristic_kernel_weights(state['size_fraction'], state['spread_complexity'], state['crowding_complexity'])
        kernel_prior_error = (kernel_weights - kernel_prior).pow(2).mean() if self.kernel_weight_mode == 'learned_adaptive' else reconstruction.new_tensor(0.0)
        dist_loss = set_mmd(target, reconstruction, weights=kernel_weights)
        moment_loss = moment_alignment(target, reconstruction)
        anchor_loss = self._conditional_anchor_mmd(target, reconstruction, kernel_weights)
        dist_scale_factor = reconstruction.new_tensor(min(1.0, float(target.shape[0]) / float(self.distribution_ramp_denom))) if self.distribution_ramp_denom > 0.0 else reconstruction.new_tensor(1.0)
        augmented_state = self._encode_state(self._augment_view(target), sample=False)
        subset_view, removed_view = self._subset_split(target)
        subset_state = self._encode_state(subset_view, sample=False)
        removed_state = self._encode_state(removed_view, sample=False)
        negative_state = self._encode_state(self._negative_view(target), sample=False)
        semantic_cosine = torch.sum(state['semantic_embedding'] * augmented_state['semantic_embedding'])
        subset_cosine = torch.sum(state['semantic_embedding'] * subset_state['semantic_embedding'])
        negative_cosine = torch.sum(state['semantic_embedding'] * negative_state['semantic_embedding'])
        queue_negatives = self._get_semantic_queue().to(target.device)
        shared_negatives = torch.cat([negative_state['semantic_embedding'].detach().unsqueeze(0), queue_negatives], dim=0)
        gate_features = torch.stack([state['size_fraction'], state['complexity_target'], state['crowding_complexity'], (1.0 - semantic_cosine.detach()).clamp(0.0, 1.0), (1.0 - subset_cosine.detach()).clamp(0.0, 1.0), negative_cosine.detach().clamp(0.0, 1.0)])
        dyn_gate = self.similarity_gate(gate_features.unsqueeze(0)).squeeze(0).squeeze(0)
        dyn_gate = self.dynamic_gate_floor + (self.dynamic_gate_ceiling - self.dynamic_gate_floor) * dyn_gate
        dyn_gate_target = self.dynamic_gate_floor + (self.dynamic_gate_ceiling - self.dynamic_gate_floor) * (0.45 * state['complexity_target'].detach() + 0.35 * negative_cosine.detach().clamp(0.0, 1.0) + 0.2 * (1.0 - subset_cosine.detach()).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        dyn_gate_reg = (dyn_gate - dyn_gate_target).pow(2)
        sem_loss = reconstruction.new_tensor(0.0)
        sub_loss = reconstruction.new_tensor(0.0)
        neg_loss = reconstruction.new_tensor(0.0)
        margin_loss = reconstruction.new_tensor(0.0)
        if self.semantic_mode != 'off' and float(sem_w.item()) > 0.0:
            dyn_w = sem_w * (1.0 - self.dynamic_similarity_scale + self.dynamic_similarity_scale * dyn_gate)
            infonce_aug = info_nce_loss(state['semantic_query'], augmented_state['semantic_embedding'].detach(), shared_negatives, self.semantic_temperature)
            infonce_sub = info_nce_loss(state['semantic_query'], subset_state['semantic_embedding'].detach(), shared_negatives, self.semantic_temperature)
            sem_loss = sem_w * (0.65 * infonce_aug + 0.35 * infonce_sub)
            sub_loss = sem_w * (torch.relu(0.94 - subset_cosine) + 0.5 * torch.relu(0.9 - semantic_cosine))
            neg_loss = dyn_w * torch.relu(negative_cosine - self.neg_cosine_threshold)
            margin_loss = dyn_w * torch.relu(self.margin_gap - (semantic_cosine - negative_cosine))
        repulsion_loss = -torch.log((1.0 - negative_cosine).clamp_min(0.0001))
        raw_projs = torch.stack([state['raw_semantic_projection'], augmented_state.get('raw_semantic_projection', state['raw_semantic_projection']), subset_state.get('raw_semantic_projection', state['raw_semantic_projection']), negative_state.get('raw_semantic_projection', state['raw_semantic_projection'])], dim=0)
        raw_std = raw_projs.std(dim=0)
        prenorm_var_loss = torch.relu(self.prenorm_var_target - raw_std).mean()
        batch_std = raw_std
        mono_loss = torch.relu(subset_state['slot_budget'] - state['slot_budget'] + 0.02)
        hys_latent = (state['combined_latent'] - subset_state['combined_latent'] - removed_state['combined_latent']).pow(2).mean()
        hys_budget = (state['slot_budget'] - subset_state['slot_budget'] - removed_state['slot_budget']).pow(2)
        hys_recon = reconstruction.new_tensor(0.0)
        hys_subset_recon = reconstruction.new_tensor(0.0)
        hys_remainder_recon = reconstruction.new_tensor(0.0)
        hys_slot = reconstruction.new_tensor(0.0)
        hys_gate = reconstruction.new_tensor(0.0)
        hys_refresh = reconstruction.new_tensor(0.0)
        hys_objective_on = self.hysteresis_recon_scale > 0.0 or self.hysteresis_slot_scale > 0.0 or self.hysteresis_gate_scale > 0.0
        if self.use_hvq and hys_objective_on:
            hys_subset_view, hys_removed_view = self._subset_split(x)
            hys_subset_state = self._encode_state(hys_subset_view, sample=False)
            hys_removed_state = self._encode_state(hys_removed_view, sample=False)
        if self.use_hvq and hys_objective_on and (hys_subset_state['target'].shape[0] > 0):
            subset_dec = self._decode_tacd_autoencoder(hys_subset_state, int(hys_subset_state['target'].shape[0]))
            hys_subset_recon = mse_between_sets(hys_subset_state['target'], subset_dec['reconstruction'], 'hungarian')
            remainder_state = self._hysteresis_remainder_state(state, hys_subset_state, hys_removed_state)
            hys_refresh = remainder_state.get('hysteresis_refresh', hys_refresh)
            remainder_dec = self._decode_tacd_autoencoder(remainder_state, int(hys_subset_state['target'].shape[0]))
            hys_remainder_recon = mse_between_sets(hys_subset_state['target'], remainder_dec['reconstruction'], 'hungarian')
            hys_recon = 0.35 * hys_subset_recon + 0.65 * hys_remainder_recon
            full_slots = state.get('hvq_fine_slots_prequant', state['z_slots'])
            subset_slots = hys_subset_state.get('hvq_fine_slots_prequant', hys_subset_state['z_slots'])
            removed_slots = hys_removed_state.get('hvq_fine_slots_prequant', hys_removed_state['z_slots'])
            hys_slot = (full_slots - subset_slots - removed_slots).pow(2).mean()
            hys_gate = (state['slot_gates'] - hys_subset_state['slot_gates'] - hys_removed_state['slot_gates']).pow(2).mean()
        detail_penalty = torch.relu(detail_loss - coarse_loss.detach() + 0.01)
        struct_w = reconstruction.new_tensor(0.3 + 0.7 * min(1.0, progress / 0.15))
        phase_b_w = reconstruction.new_tensor(min(1.0, max(0.0, (progress - self.phase_a_end) / (self.phase_b_end - self.phase_a_end))) if progress >= self.phase_a_end else 0.0)
        dist_w = sem_w
        extra_loss = struct_w * self.s.budget * budget_loss + struct_w * self.s.entropy * entropy_loss + struct_w * self.s.active * active_slot_loss + struct_w * self.s.complexity * complexity_loss + coarse_sup_w * coarse_loss + phase_b_w * self.detail_loss_scale * detail_loss + phase_b_w * 0.2 * detail_penalty + phase_b_w * self.hvq_loss_scale * hvq_loss + phase_b_w * self.hvq_entropy_scale * hvq_usage_loss + phase_b_w * hvq_residual_reg + phase_b_w * self.hvq_variable_capacity_loss_scale * hvq_capacity_rate_loss + struct_w * self.s.monotonicity * mono_loss + (0.5 + 0.5 * sem_w) * self.s.hysteresis * hys_latent + (0.5 + 0.5 * sem_w) * self.s.hysteresis_budget * hys_budget + phase_b_w * self.hysteresis_recon_scale * hys_recon + phase_b_w * self.hysteresis_slot_scale * hys_slot + phase_b_w * self.hysteresis_gate_scale * hys_gate + dist_w * self.s.dist * dist_scale_factor * dist_loss + dist_w * self.s.moment * dist_scale_factor * moment_loss + dist_w * self.s.anchor * dist_scale_factor * anchor_loss + dist_w * self.s.kernel_prior * kernel_prior_error + self.s.semantic * sem_loss + self.s.subset * sub_loss + self.s.negative * neg_loss + self.s.margin * margin_loss + sem_w * self.s.gate_reg * dyn_gate_reg + sem_w * self.repulsion_scale * repulsion_loss + sem_w * self.prenorm_var_scale * prenorm_var_loss
        self._enqueue_semantic(state['semantic_embedding'].detach())
        self.last_aux = {'kl': kl_coarse + struct_w * kl_slots, 'extra_loss': extra_loss, 'stats': {'slot_budget': float(state['slot_budget'].item()), 'soft_slot_budget': float(state.get('hvq_soft_slot_budget', state['slot_budget']).item()), 'budget_floor_applied': float(state.get('budget_floor_applied', torch.tensor(0.0)).item()), 'budget_target': float(budget_target.item()), 'count_hint': float(state['count_hint'].item()), 'complexity_hint': float(state['complexity_hint'].item()), 'complexity_target': float(state['complexity_target'].item()), 'complexity_level': float(state['complexity_level'].item()), 'raw_complexity_level': float(state['raw_complexity_level'].item()), 'spread_complexity': float(state['spread_complexity'].item()), 'crowding_complexity': float(state['crowding_complexity'].item()), 'size_prior': float(state['size_prior'].item()), 'size_fraction': float(state['size_fraction'].item()), 'slot_entropy': float(slot_entropy.item()), 'active_slots@0.25': float((state['slot_gates'] > 0.25).float().sum().item()), 'hvq_active_slots': float(state.get('hvq_active_slots', reconstruction.new_tensor(0.0)).item()), 'hvq_target_active_slots': float(state.get('hvq_target_active_slots', active_target).item()), 'hvq_capacity_signal': float(state.get('hvq_capacity_signal', state['complexity_target']).item()), 'hvq_capacity_rate_loss': float(hvq_capacity_rate_loss.item()), 'hvq_variable_capacity': float(state.get('hvq_variable_capacity_enabled', reconstruction.new_tensor(0.0)).item()), 'hvq_variable_capacity_loss_scale': float(self.hvq_variable_capacity_loss_scale), 'soft_active_slots': float(soft_active_slots.item()), 'active_target': float(active_target.item()), 'semantic_cosine': float(semantic_cosine.item()), 'subset_cosine': float(subset_cosine.item()), 'negative_cosine': float(negative_cosine.item()), 'blend_weight': float(blend_w.item()), 'phase_b_weight': float(phase_b_w.item()), 'semantic_weight': float(sem_w.item()), 'coarse_sup_weight': float(coarse_sup_w.item()), 'structure_weight': float(struct_w.item()), 'dynamic_similarity_gate': float(dyn_gate.item()), 'dynamic_gate_target': float(dyn_gate_target.item()), 'dynamic_gate_reg': float(dyn_gate_reg.item()), 'semantic_temperature': float(self.semantic_temperature), 'semantic_queue_count': float(self.semantic_queue_count.item()), 'subset_budget': float(subset_state['slot_budget'].item()), 'dist_mmd': float(dist_loss.item()), 'distribution_scale': float(dist_scale_factor.item()), 'moment_alignment': float(moment_loss.item()), 'anchor_mmd': float(anchor_loss.item()), 'hysteresis_latent': float(hys_latent.item()), 'hysteresis_budget': float(hys_budget.item()), 'hysteresis_recon': float(hys_recon.item()), 'hysteresis_subset_recon': float(hys_subset_recon.item()), 'hysteresis_remainder_recon': float(hys_remainder_recon.item()), 'hysteresis_slot': float(hys_slot.item()), 'hysteresis_gate': float(hys_gate.item()), 'hysteresis_refresh': float(hys_refresh.item()), 'hysteresis_recon_scale': float(self.hysteresis_recon_scale), 'hysteresis_slot_scale': float(self.hysteresis_slot_scale), 'hysteresis_gate_scale': float(self.hysteresis_gate_scale), 'hysteresis_refresh_blend': float(self.hysteresis_refresh_blend), 'kernel_weight_local': float(kernel_weights[0].item()), 'kernel_weight_medium': float(kernel_weights[1].item()), 'kernel_weight_broad': float(kernel_weights[2].item()), 'kernel_weight_global': float(kernel_weights[3].item()), 'kernel_prior_error': float(kernel_prior_error.item()), 'mixture_entropy': float(state['mixture_entropy'].item()), 'mixture_weight_peak': float(state['mixture_weights'].max().item()), 'coarse_loss': float(coarse_loss.item()), 'detail_loss': float(detail_loss.item()), 'final_loss': float(final_loss.item()), 'detail_delta': float(detail_delta.item()), 'detail_penalty': float(detail_penalty.item()), 'hvq_enabled': float(1.0 if self.use_hvq else 0.0), 'hvq_num_codes': float(self.hvq_num_codes), 'hvq_loss': float(hvq_loss.item()), 'hvq_commitment_loss': float(state.get('hvq_commitment_loss', reconstruction.new_tensor(0.0)).item()), 'hvq_codebook_loss': float(state.get('hvq_codebook_loss', reconstruction.new_tensor(0.0)).item()), 'hvq_usage_loss': float(hvq_usage_loss.item()), 'hvq_residual_reg': float(hvq_residual_reg.item()), 'hvq_perplexity': float(state.get('hvq_perplexity', reconstruction.new_tensor(0.0)).item()), 'hvq_active_codes': float(state.get('hvq_active_codes', 0)), 'hvq_bandwidth_gate': float(state.get('hvq_bandwidth_gate', reconstruction.new_tensor(0.0)).item()), 'hvq_fine_gate': float(fine_gate_mean.item()), 'hvq_fine_residual_norm': float(fine_residual_norm.item()), 'hvq_outlier_strength': float(state.get('hvq_outlier_strength', reconstruction.new_tensor(0.0)).item()), 'hvq_outlier_active': float(state.get('hvq_outlier_mask', reconstruction.new_zeros(self.num_slots)).sum().item()), 'hvq_outlier_injection_scale': float(self.hvq_outlier_injection_scale), 'coarse_dim': float(self.coarse_dim), 'coarse_refine_gain': float(torch.tanh(self.coarse_refine_gain).item()), 'detail_refine_gain': float(torch.tanh(self.detail_refine_gain).item()), 'refine_gain': float(torch.tanh(self.refine_gain).item()), 'progress': float(progress), 'repulsion_loss': float(repulsion_loss.item()), 'prenorm_var_loss': float(prenorm_var_loss.item()), 'raw_proj_std_mean': float(batch_std.mean().item()), 'raw_proj_std_min': float(batch_std.min().item())}}
        self.last_aux_target = target.detach()
        self.last_aux_reconstruction = reconstruction.detach()
        return AutoencoderOutput(reconstruction=reconstruction, latent=state['combined_latent'], predicted_count=predicted_count, target=target, decode_count=int(decode_count))


class TACDTaskConditioned(TACDBackbone):
    """Backbone with task embedding and optional behavior head."""

    def __init__(self, *, num_tasks: int=2, action_dim: int=2, use_real_tau: bool=True, use_behavior_head: bool=True, task_condition_scale: float=0.5, total_steps_hint: int=500, **kwargs):
        task_mode = 'learned' if use_real_tau else 'off'
        super().__init__(task_condition_mode=task_mode, task_condition_scale=task_condition_scale, total_steps_hint=total_steps_hint, **kwargs)
        self.num_tasks = num_tasks
        self.use_real_tau = use_real_tau
        self.use_behavior_head = use_behavior_head
        self._external_tau: int = 0
        self.task_bins = num_tasks
        latent_dim = self.task_embedding.embedding_dim
        self.task_embedding = nn.Embedding(num_tasks, latent_dim)
        hidden_dim = 128
        _advance_initialization_rng(latent_dim, hidden_dim, hidden_dim, num_tasks)
        new_sem_in = latent_dim * 2 + self.num_slots + 2 + num_tasks
        self.semantic_input_dim = new_sem_in
        self.semantic_input_norm = nn.LayerNorm(new_sem_in)
        semantic_dim = self.semantic_projector[-1].out_features
        self.semantic_projector = nn.Sequential(nn.Linear(new_sem_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, semantic_dim))
        self.policy_cond_tau = False
        if use_behavior_head:
            self.agent_policy = _task_mlp(self.backbone.input_dim, hidden_dim, action_dim, depth=2)
            self.agent_policy_cond = _task_mlp(self.backbone.input_dim + num_tasks, hidden_dim, action_dim, depth=2)
        self.action_dim = action_dim

    def set_tau(self, tau: int) -> None:
        """Set the external task ID for the next forward() call."""
        self._external_tau = int(tau)

    def _synthetic_task_id(self, complexity_stats: dict) -> torch.Tensor:
        """Return the real task ID or the backbone heuristic."""
        if self.use_real_tau:
            ref = complexity_stats['size_fraction']
            return ref.new_tensor(self._external_tau, dtype=torch.long)
        raw = super()._synthetic_task_id(complexity_stats)
        return raw.clamp(max=self.num_tasks - 1)

    def policy_actions(self, x_hat: torch.Tensor, tau: int | None=None) -> torch.Tensor:
        """Apply the policy head to each reconstructed agent state."""
        if self.policy_cond_tau and tau is not None:
            onehot = F.one_hot(x_hat.new_tensor(int(tau), dtype=torch.long), num_classes=self.num_tasks).float()
            onehot = onehot.unsqueeze(0).expand(x_hat.shape[0], -1)
            feat = torch.cat([x_hat, onehot], dim=-1)
            return self.agent_policy_cond(feat)
        return self.agent_policy(x_hat)


    def _decode_tacd_autoencoder(self, state, decode_count):
        """Decode the coarse, detail and fine-memory representations."""
        if decode_count <= 0:
            return super()._decode_tacd_autoencoder(state, decode_count)
        coarse_base = self.backbone.decode(state['coarse_latent'], decode_count)
        coarse_ctx = torch.cat([coarse_base, state['coarse_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
        coarse_residual = torch.tanh(self.coarse_refine_gain) * self.coarse_refiner(coarse_ctx)
        refined = coarse_base + coarse_residual
        if self.coarse_refine_layers >= 2:
            ctx2 = torch.cat([refined, state['coarse_latent'].unsqueeze(0).expand(decode_count, -1), coarse_base], dim=-1)
            coarse_residual_2 = torch.tanh(self.coarse_refine_gain_2) * self.coarse_refiner_2(ctx2)
            coarse_recon = refined + coarse_residual_2
        else:
            coarse_recon = refined
        detail_base = self.backbone.decode(state['combined_latent'], decode_count)
        detail_ctx = torch.cat([detail_base, state['combined_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
        detail_residual = torch.tanh(self.detail_refine_gain) * self.detail_refiner(detail_ctx)
        detail_recon = detail_base + detail_residual
        fine_residual_norm = state['coarse_latent'].new_tensor(0.0)
        fine_gate_mean = state['coarse_latent'].new_tensor(0.0)
        if getattr(self, 'use_hvq', False):
            queries = self._hvq_decode_queries(decode_count, state['coarse_latent'].device, state['coarse_latent'].dtype)
            attn_scores = queries @ state['z_slots'].transpose(0, 1) / float(self.slot_dim) ** 0.5
            attn_scores = attn_scores + torch.log(state['slot_gates'].clamp_min(0.0001)).unsqueeze(0)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            fine_context = attn_weights @ state['z_slots']
            object_gate = (attn_weights * state['slot_gates'].unsqueeze(0)).sum(dim=-1, keepdim=True)
            fine_ctx = torch.cat([detail_recon, fine_context, state['combined_latent'].unsqueeze(0).expand(decode_count, -1)], dim=-1)
            fine_residual = torch.tanh(self.residual_decoder(fine_ctx))
            fine_gate = state['hvq_bandwidth_gate'] * object_gate
            detail_recon = detail_recon + self.hvq_fine_residual_scale * fine_gate * torch.tanh(self.residual_gain) * fine_residual
            fine_residual_norm = fine_residual.pow(2).mean()
            fine_gate_mean = fine_gate.mean()
        blend = self._blend_weight()
        blend_t = state['coarse_latent'].new_tensor(blend)
        reconstruction = (1.0 - blend_t) * coarse_recon + blend_t * detail_recon
        return {'reconstruction': reconstruction, 'coarse_recon': coarse_recon, 'detail_recon': detail_recon, 'blend_weight': blend_t, 'fine_residual_norm': fine_residual_norm, 'fine_gate_mean': fine_gate_mean}


class TACDModel(TACDTaskConditioned):
    """Task-conditioned TACD model used by the ablation sweep."""
    asym_flip_dims: tuple[int, int] = (2, 4)

    def _apply_asym(self, target: torch.Tensor) -> torch.Tensor:
        tau = int(getattr(self, '_external_tau', 0))
        if tau == 1:
            target = target.clone()
            lo, hi = self.asym_flip_dims
            target[..., lo:hi] = -target[..., lo:hi]
        return target

    def _encode_state(self, x: torch.Tensor, sample: bool):
        state = super()._encode_state(x, sample)
        if 'target' in state:
            state['target'] = self._apply_asym(state['target'])
        return state

    def __init__(self, *, training_max_agents: int=12, **kwargs):
        super().__init__(**kwargs)
        _advance_initialization_rng(self.task_embedding.embedding_dim, 64, 2 * self.backbone.input_dim)
        _advance_initialization_rng(self.backbone.input_dim, 64, self.num_tasks)
        if not isinstance(training_max_agents, int) or not 1 <= training_max_agents <= self.backbone.max_n:
            raise ValueError('training_max_agents must be an integer within the embedding range')
        self.training_max_agents = training_max_agents
        self.execution_stage = 'train'
        self.backbone.training_max_agents = training_max_agents
        self.backbone.execution_stage = 'train'

    def set_execution_stage(self, stage: str):
        self.backbone.set_execution_stage(stage)
        self.execution_stage = stage
        return self
