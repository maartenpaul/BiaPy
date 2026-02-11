# Experiment Tracking (MLflow & Weights & Biases)

BiaPy supports logging training metrics, hyperparameters, and artifacts to
**TensorBoard** (default), **Weights & Biases (wandb)**, and **MLflow**.

## Installation

wandb and mlflow are optional dependencies. Install the one you need:

```bash
pip install biapy[wandb]      # Weights & Biases
pip install biapy[mlflow]     # MLflow
pip install biapy[all-loggers] # both
```

## Configuration

Add a `LOG` section to your YAML config file. TensorBoard remains the default
when nothing is specified, so existing configs keep working unchanged.

### Weights & Biases

```yaml
LOG:
  LOGGER: "WANDB"
  LOG_ARTIFACTS: true       # optional, default false
  WANDB:
    PROJECT: "my-project"   # defaults to job name if empty
    ENTITY: "my-team"       # optional
    TAGS: ["3D", "unet"]    # optional
    GROUP: ""               # optional, groups related runs
    SAVE_DIR: ""            # optional, local dir for wandb files
```

### MLflow

```yaml
LOG:
  LOGGER: "MLFLOW"
  LOG_ARTIFACTS: true
  MLFLOW:
    TRACKING_URI: "http://localhost:5000"  # empty = local ./mlruns
    EXPERIMENT_NAME: "cell-segmentation"   # defaults to job name if empty
```

### All loggers at once

```yaml
LOG:
  LOGGER: "ALL"
```

### Disable all logging

```yaml
LOG:
  LOGGER: "NONE"
```

## What gets logged

### Metrics (automatic)

Every training run logs the following scalars at each step/epoch:

| Metric | Tag | When |
|--------|-----|------|
| Training loss | `loss/loss` | Every training step |
| Learning rate | `opt/lr` | Every training step |
| Validation loss | `perf/test_loss` | End of each epoch |
| Task metric (e.g. IoU) | `perf/test_<metric>` | End of each epoch |

These are logged identically to what TensorBoard already receives — no extra
configuration needed.

### Hyperparameters (automatic)

The full BiaPy configuration dictionary is logged as hyperparameters at the
start of each training run:

- **wandb**: stored in `wandb.config` (visible in the run's config tab)
- **MLflow**: stored as MLflow params (flattened with dot-separated keys,
  e.g. `TRAIN.LR`, `MODEL.ARCHITECTURE`)

### Artifacts (opt-in via `LOG_ARTIFACTS: true`)

When `LOG.LOG_ARTIFACTS` is enabled, the following files are uploaded:

| Artifact | When | Description |
|----------|------|-------------|
| Periodic checkpoint | Every `MODEL.SAVE_CKPT_FREQ` epochs | Full model checkpoint (`.pth` / `.safetensors`) |
| Best checkpoint | When validation loss improves | `<job>-checkpoint-best.pth` |
| Loss chart | Every `LOG.CHART_CREATION_FREQ` epochs | `<job>_loss.png` matplotlib plot |
| Metric charts | Every `LOG.CHART_CREATION_FREQ` epochs | `<job>_<metric>.png` per metric |

**Where artifacts end up:**

- **wandb**: uploaded as versioned `wandb.Artifact` objects (type `"model"`),
  visible in the run's Artifacts tab
- **MLflow**: uploaded via `mlflow.log_artifact()`, visible in the run's
  Artifacts section in the MLflow UI

Artifact uploads are wrapped in try/except — a failed upload prints a warning
but never crashes training.

## Adding to an existing BiaPy workflow

No code changes are needed. Simply add the `LOG` keys to your existing YAML
config. For example, starting from a standard semantic segmentation config:

```yaml
PROBLEM:
  TYPE: SEMANTIC_SEG
  NDIM: 2D

DATA:
  PATCH_SIZE: (256, 256, 1)
  TRAIN:
    PATH: /data/train/images
    GT_PATH: /data/train/labels

MODEL:
  ARCHITECTURE: unet

TRAIN:
  ENABLE: True
  OPTIMIZER: ADAMW
  LR: 1.E-3
  EPOCHS: 50

TEST:
  ENABLE: True

# --- add this block ---
LOG:
  LOGGER: "WANDB"
  LOG_ARTIFACTS: true
  WANDB:
    PROJECT: "cell-segmentation"
    TAGS: ["semantic_seg", "unet"]
```

Then run BiaPy as usual:

```bash
biapy --config config.yaml --result_dir ./results --name my_run --gpu 0
```

wandb will initialise automatically, and you can view metrics live in your
wandb dashboard.

### Using MLflow with a local server

```bash
# Terminal 1: start MLflow UI
mlflow ui --port 5000

# Terminal 2: run BiaPy
biapy --config config.yaml --result_dir ./results --name my_run --gpu 0
```

With `TRACKING_URI: "http://localhost:5000"` in your config, metrics and
artifacts will appear in the MLflow UI at http://localhost:5000.

## Available `LOG.LOGGER` values

| Value | Backends enabled |
|-------|-----------------|
| `"TENSORBOARD"` | TensorBoard only (default) |
| `"WANDB"` | Weights & Biases only |
| `"MLFLOW"` | MLflow only |
| `"ALL"` | TensorBoard + wandb + MLflow |
| `"NONE"` | No experiment tracking |

## Architecture note

The integration uses a `BaseLogger` abstract class with a `CompositeLogger`
that dispatches calls to one or more backends. The training loop code
(`train_engine.py`, `base_workflow.py`) only interacts with the logger through
`set_step()`, `update()`, and `flush()` — the same interface as the original
TensorBoard logger. This means adding a new backend in the future only requires
implementing `BaseLogger` and registering it in the factory
(`biapy/utils/loggers/factory.py`).
