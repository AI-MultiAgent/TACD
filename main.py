"""TACD dataset, training, and reconstruction evaluation entry point."""
from __future__ import annotations
import argparse
import copy
import json
import platform
import random
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model import TACDModel

ROOT = Path(__file__).resolve().parent
INPUT_DIM, ACTION_DIM, NUM_TASKS = 6, 2, 2
LATENT_DIM, MAX_N, STEPS = 96, 100, 1500
BEHAVIOR_WEIGHT, POSITION_ANCHOR_WEIGHT = 1.0, 0.05
BEHAVIOR_WARMUP_STEPS, BEHAVIOR_RAMP_STEPS = 80, 60
OUTPUT_STABILITY_WEIGHT, OUTPUT_STABILITY_BOUND, GRAD_CLIP_NORM = 0.02, 3.0, 5.0
DA_BANDWIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0)
DA_LAMBDA = 0.1
VARIANTS = {
    "full": {"hvq": True, "tau": True, "da": True},
    "without_ca": {"hvq": False, "tau": True, "da": True},
    "without_ta": {"hvq": True, "tau": False, "da": True},
    "without_da": {"hvq": True, "tau": True, "da": False},
}
DATASETS = {"f1": (17, 29, 68, 71, 97), "spread": (17, 29, 68, 71, 97), "road": (17, 29, 68, 71, 97)}
SPLITS = ("IID", "OOD_20", "OOD_50", "OOD_100")
TOKENS = dict(IID=7, OOD_20=20, OOD_50=50, OOD_100=100)

CA_ADAPTIVE_KWARGS = {'coarse_dim_ratio': 8.0 / 96.0, 'num_slots': 8, 'slot_dim': 16, 'hvq_bandwidth_threshold': 0.2, 'hvq_fine_residual_scale': 1.0, 'hvq_outlier_injection_scale': 0.0, 'hvq_loss_scale': 0.05, 'hvq_entropy_scale': 0.02, 'hvq_residual_reg_scale': 0.04, 'hvq_variable_capacity': False, 'hysteresis_recon_scale': 0.0, 'hysteresis_slot_scale': 0.0, 'hysteresis_gate_scale': 0.0, 'hysteresis_refresh_blend': 0.65}

def cond_kwargs(scale=0.5):
    return dict(use_real_tau=True, use_behavior_head=True, task_condition_scale=scale)

def preserve_paper_action_order():
    return True

def clean_mmd_multiscale(x, y, bandwidths=DA_BANDWIDTHS):
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_tensor(0.0)
    sqd_xx = torch.cdist(x, x, p=2.0).pow(2)
    sqd_yy = torch.cdist(y, y, p=2.0).pow(2)
    sqd_xy = torch.cdist(x, y, p=2.0).pow(2)
    loss = x.new_tensor(0.0)
    for bw in bandwidths:
        s2 = 2.0 * float(bw) * float(bw)
        loss = loss + (torch.exp(-sqd_xx / s2).mean() + torch.exp(-sqd_yy / s2).mean() - 2.0 * torch.exp(-sqd_xy / s2).mean())
    return loss / len(bandwidths)

def _disable_backbone_distribution_family(m):
    m.s.dist = 0.0
    m.s.moment = 0.0
    m.s.anchor = 0.0
    m.s.kernel_prior = 0.0

def _disable_hvq(m):
    m.use_hvq = False
    m.hvq_loss_scale = 0.0
    m.hvq_entropy_scale = 0.0
    m.hvq_fine_residual_scale = 0.0
    m.hvq_residual_reg_scale = 0.0
    if hasattr(m, 'hysteresis_recon_scale'):
        m.hysteresis_recon_scale = 0.0
    if hasattr(m, 'hysteresis_slot_scale'):
        m.hysteresis_slot_scale = 0.0
    if hasattr(m, 'hysteresis_gate_scale'):
        m.hysteresis_gate_scale = 0.0
    if hasattr(m, 'hysteresis_refresh_blend'):
        m.hysteresis_refresh_blend = 0.0
    if hasattr(m, 'hvq_codebook'):
        for p in m.hvq_codebook.parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        m.coarse_refine_gain.fill_(0.0)
        if hasattr(m, 'coarse_refine_gain_2'):
            m.coarse_refine_gain_2.fill_(0.0)
        m.refine_gain.fill_(2.5)
    m.coarse_refine_gain.requires_grad_(False)
    if hasattr(m, 'coarse_refine_gain_2'):
        m.coarse_refine_gain_2.requires_grad_(False)
    m.refine_gain.requires_grad_(False)
    m.coarse_sup_peak = 0.0
    m.coarse_sup_floor = 0.0

def make_model(flags):
    mkw = cond_kwargs(0.5)
    mkw['use_real_tau'] = flags['tau']
    mkw['use_hvq'] = flags['hvq']
    mkw['coarse_dim_ratio'] = 0.25 if flags['hvq'] else 0.75
    if flags['hvq']:
        mkw.update(CA_ADAPTIVE_KWARGS)
    m = TACDModel(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, max_n=MAX_N, num_tasks=NUM_TASKS, action_dim=ACTION_DIM, total_steps_hint=STEPS, **mkw)
    m.policy_cond_tau = False
    m._policy_mode = 'x_hat_strict'
    _disable_backbone_distribution_family(m)
    if not flags['hvq']:
        _disable_hvq(m)
    return m

def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def hungarian_mse(target, recon):
    if target.shape[0] == 0 or recon.shape[0] == 0:
        return target.new_tensor(0.0)
    row, col = hungarian_match(target, recon)
    return F.mse_loss(recon[col], target[row])

def actions_in_target_order(model, source, actions):
    if preserve_paper_action_order():
        return actions
    if source.shape[0] != actions.shape[0]:
        raise ValueError('source observations and action labels must have equal row counts')
    return actions[model.backbone.canonical_order(source)]

def matched_position_mse(target, recon):
    if target.shape[0] == 0 or recon.shape[0] == 0:
        return target.new_tensor(0.0)
    row, col = hungarian_match(target, recon)
    return F.mse_loss(recon[col, :2], target[row, :2])

def behavior_warmup(step: int) -> float:
    if step < BEHAVIOR_WARMUP_STEPS:
        return 0.0
    return min(1.0, (step - BEHAVIOR_WARMUP_STEPS) / max(1, BEHAVIOR_RAMP_STEPS))

def hungarian_match(target: torch.Tensor, pred: torch.Tensor):
    with torch.no_grad():
        target_pos = target[:, :2] if target.shape[1] >= 2 else target
        pred_pos = pred[:, :2] if pred.shape[1] >= 2 else pred
        cost = torch.cdist(target_pos, pred_pos).cpu().numpy()
        row, col = linear_sum_assignment(cost)
    return (row, col)

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
    if matching == 'direct':
        return _pad_direct(target, prediction)
    if matching == 'hungarian':
        return _pad_hungarian(target, prediction)
    raise ValueError(f'Unsupported matching mode: {matching}')

def mse_between_sets(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> torch.Tensor:
    aligned_target, aligned_prediction = align_sets(target, prediction, matching)
    if aligned_target.numel() == 0:
        return aligned_target.new_tensor(0.0)
    return torch.mean((aligned_prediction - aligned_target) ** 2)

def mae_between_sets(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> torch.Tensor:
    aligned_target, aligned_prediction = align_sets(target, prediction, matching)
    if aligned_target.numel() == 0:
        return aligned_target.new_tensor(0.0)
    return torch.mean(torch.abs(aligned_prediction - aligned_target))

def correlation_coefficient(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> float:
    aligned_target, aligned_prediction = align_sets(target, prediction, matching)
    if aligned_target.numel() < 2:
        return 1.0
    a_flat = aligned_target.reshape(-1).float()
    b_flat = aligned_prediction.reshape(-1).float()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = a_centered.norm() * b_centered.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((a_centered @ b_centered / denom).item())

def gaussian_mmd(x: torch.Tensor, y: torch.Tensor, bandwidths: Iterable[float]=(0.25, 0.5, 1.0, 2.0, 4.0)) -> float:
    if x.shape[0] == 0 and y.shape[0] == 0:
        return 0.0
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 1.0

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sqdist = torch.cdist(a, b, p=2.0).pow(2)
        vals = [torch.exp(-sqdist / (2.0 * float(bw) * float(bw))) for bw in bandwidths]
        return sum(vals) / len(vals)
    with torch.no_grad():
        k_xx = kernel(x, x).mean()
        k_yy = kernel(y, y).mean()
        k_xy = kernel(x, y).mean()
        return float((k_xx + k_yy - 2.0 * k_xy).item())

def train_one(name, flags, train_data, val_data, seed, steps=1500, learning_rate=0.0015):
    set_seed(seed)
    model = make_model(flags)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    beta, slw, elw = (0.012, 0.05, 1.0)
    lam = DA_LAMBDA if flags['da'] else 0.0
    hist = {'step': [], 'recon': [], 'val': [], 'behavior': [], 'stability': [], 'position_anchor': []}
    for step in range(steps):
        x, actions, n, tid, _ = train_data[step % len(train_data)]
        model.set_tau(tid)
        out = model(x, decode_mode='target')
        target = out.target
        recon = out.reconstruction
        target_actions = actions_in_target_order(model, x, actions)
        rloss = hungarian_mse(target, recon)
        sloss = (out.predicted_count.view(1) - x.new_tensor(float(n)).view(1)).pow(2).mean()
        aux = model.last_aux
        da_loss = lam * clean_mmd_multiscale(target, recon) if lam > 0 else x.new_tensor(0.0)
        position_anchor_loss = POSITION_ANCHOR_WEIGHT * matched_position_mse(target, recon) if flags['hvq'] and POSITION_ANCHOR_WEIGHT > 0 else x.new_tensor(0.0)
        stability_loss = OUTPUT_STABILITY_WEIGHT * torch.relu(recon.abs() - OUTPUT_STABILITY_BOUND).pow(2).mean()
        behav_loss = x.new_tensor(0.0)
        if flags['tau'] and model.use_behavior_head:
            pred_actions = model.policy_actions(recon, tau=None)
            row, col = hungarian_match(target, recon)
            behav_loss = behavior_warmup(step) * F.mse_loss(pred_actions[col], target_actions[row])
        total = rloss + slw * sloss + beta * aux['kl'] + elw * aux['extra_loss'] + da_loss + BEHAVIOR_WEIGHT * behav_loss + stability_loss + position_anchor_loss
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                v, nv = (0.0, 0)
                for vs in val_data[:25]:
                    vx, _va, vn, vt, _ = vs
                    model.set_tau(vt)
                    vo = model(vx, decode_mode='target')
                    v += float(hungarian_mse(vo.target, vo.reconstruction).item())
                    nv += 1
                v /= max(nv, 1)
            hist['step'].append(step)
            hist['recon'].append(float(rloss.item()))
            hist['val'].append(v)
            hist['behavior'].append(float(behav_loss.item()))
            hist['stability'].append(float(stability_loss.item()))
            hist['position_anchor'].append(float(position_anchor_loss.item()))
            if step % 200 == 0:
                print(f'  [{name}] step {step:4d} | recon {rloss.item():.4f} | da {da_loss.item():.4f} | behav {behav_loss.item():.4f} | pos {position_anchor_loss.item():.4f} | stab {stability_loss.item():.4f} | val {v:.4f}')
    return (model, hist, opt)

def load_rows(dataset, seed, split):
    path = ROOT / "datasets" / dataset / f"seed_{seed}" / f"{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    rows = []
    for index, count in enumerate(arrays["n"]):
        n = int(count)
        x = torch.from_numpy(arrays["x"][index, :n].copy())
        actions = torch.from_numpy(arrays["actions"][index, :n].copy())
        positions = (arrays["positions"][index, :n].copy() if "positions" in arrays
                     else x[:, :2].numpy().copy())
        if not np.array_equal(positions, x[:, :2].numpy()):
            raise ValueError(f"Position mismatch in {path.name}, row {index}")
        rows.append((x, actions, n, int(arrays["task_id"][index]), positions))
    expected = dict(train=400, val=80).get(split, 12)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} scenes, found {len(rows)}")
    return rows


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def save_checkpoint(path, model, optimizer, dataset, variant, seed, steps):
    payload = dict(state_dict=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone()),
        dataset=dataset, variant=variant, seed=seed, actual_adam_steps=steps,
        training_max_agents=12, latent_dim=LATENT_DIM, max_n=MAX_N,
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
            intra_threads=torch.get_num_threads(), inter_threads=torch.get_num_interop_threads(), device="cpu"))
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


@torch.no_grad()
def evaluate(model, dataset, seed):
    model.set_execution_stage("evaluation")
    model.eval()
    records = []
    predictions = {}
    for split in SPLITS:
        torch.manual_seed(seed * 10000 + TOKENS[split])
        for index, (x, actions, n, tau, _) in enumerate(load_rows(dataset, seed, split)):
            model.set_tau(tau)
            output = model(x, decode_mode="target")
            target, prediction = output.target, output.reconstruction
            records.append(dict(dataset=dataset, seed=seed, split=split, id=index, n=n, tau=tau,
                MSE=float(mse_between_sets(target, prediction, "hungarian").item()),
                MAE=float(mae_between_sets(target, prediction, "hungarian").item()),
                MMD=gaussian_mmd(target, prediction),
                CORR=correlation_coefficient(target, prediction, "hungarian")))
            prefix = f"{split}_{index:02d}"
            for name, value in dict(x=x, actions=actions, target=target, prediction=prediction,
                    latent=output.latent, predicted_count=output.predicted_count).items():
                predictions[f"{prefix}_{name}"] = value.detach().cpu().numpy().copy()
    return records, predictions


def rounded(value):
    return format(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), ".3f")


def show_scores(records):
    print("Regime     MSE      MAE      MMD")
    for split in SPLITS:
        rows = [row for row in records if row["split"] == split]
        if rows:
            means = [float(np.mean([row[metric] for row in rows])) for metric in ("MSE", "MAE", "MMD")]
            print(f"{split:<10} " + "  ".join(rounded(value) for value in means))


def configure_runtime(mode, threads=None):
    intra, inter = ((4, 10) if mode == "train" else (1, 1))
    torch.set_num_threads(threads if threads is not None else intra)
    torch.set_num_interop_threads(inter)


def selected_seeds(dataset, seed=None):
    if seed is None:
        return DATASETS[dataset]
    if seed not in DATASETS[dataset]:
        raise ValueError(f"Bundled {dataset} seeds are {DATASETS[dataset]}")
    return (seed,)


def preflight_input_paths(dataset, seeds, mode):
    splits = ("train", "val") if mode == "train" else SPLITS
    for seed in seeds:
        for split in splits:
            path = ROOT / "datasets" / dataset / f"seed_{seed}" / f"{split}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {path}")


def load_evaluation_checkpoints(dataset, variant, seeds, checkpoint_path=None, single_seed=False):
    if checkpoint_path is None:
        raise ValueError("evaluate requires --checkpoint pointing to a checkpoint file or run directory")
    else:
        checkpoint_path = Path(checkpoint_path).resolve()
        if checkpoint_path.is_dir():
            paths = {seed: checkpoint_path / f"seed_{seed}" / "checkpoint.pt" for seed in seeds}
        elif single_seed:
            paths = {seeds[0]: checkpoint_path}
        else:
            raise ValueError("Five-seed evaluation requires a checkpoint directory with seed_<seed>/checkpoint.pt; use --seed for a single checkpoint file")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoints = {}
    for seed, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint has no model state: {path}")
        for key, expected in dict(dataset=dataset, variant=variant, seed=seed).items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint {key} mismatch in {path}: expected {expected!r}, found {checkpoint.get(key)!r}")
        checkpoints[seed] = checkpoint
    return checkpoints


def summarize_seed_records(records_by_seed):
    metrics = ("MSE", "MAE", "MMD", "CORR")
    seeds = list(records_by_seed)
    if not seeds:
        raise ValueError("At least one seed is required for a summary")
    per_seed = []
    for seed, records in records_by_seed.items():
        regimes = {}
        for split in SPLITS:
            rows = [row for row in records if row["split"] == split]
            if len(rows) != 12 or any(row["seed"] != seed for row in rows):
                raise ValueError(f"Expected 12 scenes for seed {seed}, split {split}")
            regimes[split] = dict(n_scenes=len(rows), **{
                metric: statistics.mean(row[metric] for row in rows) for metric in metrics})
        per_seed.append(dict(seed=seed, regimes=regimes))
    regimes = {}
    for split in SPLITS:
        regimes[split] = dict(n_seeds=len(seeds), scenes_per_seed=12, **{
            metric: statistics.mean(row["regimes"][split][metric] for row in per_seed)
            for metric in metrics})
    return dict(seeds=seeds, n_seeds=len(seeds), aggregation="equal mean of per-seed scene means",
        per_seed=per_seed, regimes=regimes)


def show_seed_summary(summary):
    print(f"Mean across {summary['n_seeds']} seeds: {', '.join(map(str, summary['seeds']))}")
    print("Regime     MSE      MAE      MMD")
    for split in SPLITS:
        values = summary["regimes"][split]
        print(f"{split:<10} " + "  ".join(rounded(values[metric]) for metric in ("MSE", "MAE", "MMD")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_datasets", action="store_true", help="list bundled datasets and seeds")
    parser.add_argument("--data", "--dataset", choices=tuple(DATASETS), default="f1")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--seed", type=int, default=None, help="run one seed; omit to run all five dataset seeds")
    parser.add_argument("--steps", type=int, default=STEPS, help="optimization budget for a fresh run")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.0015)
    parser.add_argument("--checkpoint", type=Path, help="required for evaluation: single file with --seed, or a run directory containing seed_<seed>/checkpoint.pt")
    parser.add_argument("--output_dir", type=Path, help="fresh directory for new run artifacts")
    parser.add_argument("--threads", type=int, help="override CPU intra-operation threads")
    args = parser.parse_args()
    if args.list_datasets:
        for name, seeds in DATASETS.items():
            print(f"{name}: seeds={','.join(map(str, seeds))}; train=400, val=80; test=12 per regime")
        return
    try:
        seeds = selected_seeds(args.data, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.mode == "train" and args.checkpoint is not None:
        parser.error("--checkpoint is for evaluation; training starts from a fresh initialization")
    if args.output_dir is None:
        if args.mode == "train":
            parser.error("train requires --output_dir pointing to a fresh directory")
        output_dir = None
    else:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            parser.error(f"--output_dir must be a fresh directory: {output_dir}")
    try:
        preflight_input_paths(args.data, seeds, args.mode)
        checkpoints = (load_evaluation_checkpoints(args.data, args.variant, seeds, args.checkpoint,
            single_seed=args.seed is not None) if args.mode == "evaluate" else None)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    configure_runtime(args.mode, args.threads)
    flags = dict(VARIANTS[args.variant])
    print(f"{args.data} / {args.variant} / {args.mode}: seeds={','.join(map(str, seeds))}")
    batch = args.seed is None
    if args.mode == "train":
        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}" if batch else output_dir
            if batch:
                seed_dir.mkdir(exist_ok=False)
            model, history, optimizer = train_one(args.variant, flags, load_rows(args.data, seed, "train"),
                load_rows(args.data, seed, "val"), seed, args.steps, args.learning_rate)
            save_checkpoint(seed_dir / "checkpoint.pt", model, optimizer, args.data, args.variant, seed, args.steps)
            write_json(seed_dir / "history.json", history)
        if batch:
            write_json(output_dir / "summary.json", dict(mode="train", dataset=args.data, variant=args.variant,
                seeds=list(seeds), n_seeds=len(seeds), actual_adam_steps_per_seed=args.steps,
                actual_adam_steps_total=args.steps * len(seeds),
                checkpoints={str(seed): f"seed_{seed}/checkpoint.pt" for seed in seeds}))
            print(f"Saved {args.steps} optimization steps per seed across {len(seeds)} seeds to {output_dir}")
        else:
            print(f"Saved {args.steps} optimization steps to {output_dir}")
    else:
        records_by_seed = {}
        for seed in seeds:
            model = make_model(flags)
            model.load_state_dict(checkpoints[seed]["state_dict"], strict=True)
            records, predictions = evaluate(model, args.data, seed)
            records_by_seed[seed] = records
            if not batch:
                show_scores(records)
            if output_dir is not None:
                seed_dir = output_dir / f"seed_{seed}" if batch else output_dir
                if batch:
                    seed_dir.mkdir(exist_ok=False)
                write_json(seed_dir / "scores.json", records)
                np.savez_compressed(seed_dir / "predictions.npz", **predictions)
        if batch:
            summary = summarize_seed_records(records_by_seed)
            summary.update(mode="evaluate", dataset=args.data, variant=args.variant)
            show_seed_summary(summary)
            if output_dir is not None:
                write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
