# TACD

**Task-Aware and Capacity-Adaptive Distribution Alignment**

A PyTorch implementation for representing and reconstructing variable-size multi-agent state sets.

[Quick start](#quick-start) · [Model](#model) · [Evaluation](#evaluation) · [Data](#data) · [Options](#options)

## Quick start

### 1. Install

Training and evaluation use the **same Python environment**.

| Python | PyTorch | NumPy | SciPy |
|---|---|---|---|
| 3.14.3 | 2.11.0 | 2.4.3 | 1.17.1 |

Install the dependencies with:

```bash
python -m pip install torch==2.11.0 numpy==2.4.3 scipy==1.17.1
```

### 2. Train

Run commands from the repository root.

```bash
python main.py --list_datasets
python main.py --data f1 --output_dir ../tacd_runs/f1_full
```

**Defaults:** full model · 5 seeds · 1500 Adam steps per seed · learning rate 0.0015.

The output directory must be fresh. Each seed writes a checkpoint and training history under `seed_<seed>/`; `summary.json` records the batch.

### 3. Evaluate

Run reconstruction evaluation in the same environment.

```bash
python main.py --mode evaluate --data f1 --checkpoint ../tacd_runs/f1_full
```

Evaluation requires a checkpoint file or a five-seed run directory matching the dataset, variant and seed. Add `--output_dir` to save new scores and predictions to a fresh directory.

<details>
<summary>Run one seed</summary>

```bash
python main.py --data spread --seed 17 --output_dir ../tacd_runs/spread_full_17
python main.py --mode evaluate --data spread --seed 17 --checkpoint ../tacd_runs/spread_full_17/checkpoint.pt
```

A single checkpoint file requires its matching `--seed`. Omitting `--seed` runs all five dataset seeds.

</details>

## Model

| Module | Role |
|---|---|
| **CA — Capacity allocation** | Combines a compact coarse summary with gated, vector-quantized fine slots. |
| **TA — Task alignment** | Injects task embeddings and applies behavior-constrained supervision. |
| **DA — Distribution alignment** | Adds multi-scale Gaussian MMD between reconstructed and target state sets. |

`model/model.py` contains the shared backbone, CA and TA components, and model objectives. `main.py` defines the external DA objective and training/evaluation protocol. Agent-count supervision is a separate objective.

## Evaluation

Evaluation covers **IID** and **OOD-20 / 50 / 100**. All variants observe every agent and reconstruct the known number of agents.

| Metric | Measures | Direction | Reported in |
|---|---|:---:|---|
| **MSE** | Mean squared error after full-state set matching | ↓ | Console and saved scores |
| **MAE** | Mean absolute error under the same matching | ↓ | Console and saved scores |
| **MMD** | Five-bandwidth Gaussian MMD² | ↓ | Console and saved scores |
| **CORR** | Reconstruction correlation after set matching | ↑ | Saved scores |

**Example console format** — the ellipses are placeholders, not experiment values:

```text
Mean across 5 seeds: 17, 29, 68, 71, 97
Regime     MSE      MAE      MMD
IID        ...      ...      ...
OOD_20     ...      ...      ...
OOD_50     ...      ...      ...
OOD_100    ...      ...      ...
```

In a five-seed run, each regime averages 12 scenes per seed, then gives equal weight to the five seed means. A single-seed run reports its 12-scene mean. Console values use three decimal places; saved scores retain full precision.

## Data

| Dataset | `--data` |
|---|---|
| F1 | `f1` |
| MPE2 Simple Spread | `spread` |
| VMAS Road Traffic | `road` |

**Seeds:** 17, 29, 68, 71, 97. **Inputs:** 90 NPZ files, organized as:

```text
datasets/<dataset>/seed_<seed>/<split>.npz
```

<details>
<summary>Splits and array fields</summary>

| Split | Scenes per seed | Agents per scene |
|---|---:|---:|
| `train` | 400 | 3–12 |
| `val` | 80 | 3–12 |
| `IID` | 12 | 3–12 |
| `OOD_20` | 12 | 20 |
| `OOD_50` | 12 | 50 |
| `OOD_100` | 12 | 100 |

| Field | Meaning | Type |
|---|---|---|
| `x` | Padded six-dimensional state rows | float32 |
| `actions` | Padded two-dimensional action labels | float32 |
| `n` | Valid agent count | int64 |
| `task_id` | Task identifier | int64 |

For scene `i`, use `x[i, :n[i]]` and `actions[i, :n[i]]`; exclude padded rows. F1 also retains positions and source metadata; Spread and Road retain source seeds.

</details>

## Options

<details>
<summary>Command-line arguments</summary>

| Argument | Values / default |
|---|---|
| `--data`, `--dataset` | `f1`, `spread`, `road` |
| `--mode` | `train` (default), `evaluate` |
| `--variant` | `full` (default), `without_ca`, `without_ta`, `without_da` |
| `--seed` | One seed from the dataset cohort; omit for all five |
| `--steps` | 1500 Adam steps per seed |
| `--learning_rate`, `--lr` | 0.0015 |
| `--checkpoint` | Single file with `--seed`, or a five-seed run directory |
| `--output_dir` | Fresh directory for new run artifacts |
| `--threads` | Override CPU intra-operation threads |

</details>

<details>
<summary>Training and matching protocol</summary>

| Setting | Protocol |
|---|---|
| Checkpoint | Final training state |
| Validation | First 25 validation scenes at steps 0, 50, …, 1450 and 1499 |
| Training assignment | Hungarian matching on position coordinates |
| MAE / MSE evaluation | Hungarian matching on complete six-dimensional states |
| MMD | Equal mean of biased Gaussian MMD² at bandwidths 0.25, 0.5, 1, 2 and 4 |
| Execution stage | Training stage during training/validation; evaluation stage during reconstruction evaluation |

The objective combines reconstruction, agent-count prediction, representation objectives, task behavior, distribution alignment, output stability and a position anchor. Execution stage is independent of PyTorch's `train()` / `eval()` sampling mode.

`without_ca` uses its wider continuous representation; `without_ta` disables explicit task conditioning and the behavior objective; `without_da` removes the external distribution objective.

</details>
