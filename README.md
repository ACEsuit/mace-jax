# MACE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :rocket: [JAX](https://github.com/google/jax)

This repository contains a **porting** of MACE in **jax** developed by [**Philipp Benner**](https://bamescience.github.io/), **Abhijeet Gangan**, **Mario Geiger** and **Ilyes Batatia**.

## Package overview

MACE-JAX provides a JAX/Flax training stack for atomistic models, including
streaming HDF5 data loading, gin-driven configuration, distributed training,
and utilities for preprocessing and model conversion. It includes CLI entry
points for training, preprocessing, plotting metrics, and importing Torch MACE
checkpoints. Torch-to-JAX conversion is a core feature: it lets users bring in
pre-trained foundation models and fine-tune them directly in JAX. Fixed-shape
batching is supported for efficient compilation in JAX/XLA, but it is only one
part of a broader end-to-end training workflow. The HDF5 dataset format matches
Torch MACE, so the same HDF5 files can be shared across both implementations.

### Key features

- Streaming HDF5 loader with cached padding caps per split (train/valid/test)
- Fixed-shape batching driven by `n_edge` with derived node/graph padding caps
- Gin-configured model/loss/optimizer stack with CLI overrides
- Training CLI with EMA, SWA, checkpointing (including best-checkpoint), and W&B
- Optimizer schedulers (constant, exponential, piecewise, plateau) integrated via gin
- Torch-to-JAX conversion utilities and corresponding CLI
- Multi-head training/finetuning support with per-head dataset configs
- Distributed training via `jax.distributed` with per-process dataset sharding
- Preprocessing CLI to convert XYZ to streaming HDF5 plus statistics.json output

## Test without installing

```sh
pip install nox
nox
```

## Installation

From github:

```sh
pip install git+https://github.com/ACEsuit/mace-jax
```

Or locally:

```sh
pip install -e .
```

## Quick start

```sh
mkdir -p data/hdf5
mace-jax-preprocess \
  --train_file data/rmd17_aspirin_train.xyz \
  --valid_file data/rmd17_aspirin_test.xyz \
  --h5_prefix data/hdf5/ \
  --r_max 5.0 \
  --compute_statistics \
  --atomic_numbers "[1, 6, 7, 8]" \
  --E0s "average"

JAX_PLATFORMS=cpu mace-jax-train configs/aspirin_small.gin --print-config
```

This prints the operative gin configuration before training starts. On CPU-only
machines the `JAX_PLATFORMS=cpu` override avoids CUDA plugin warnings.

## Usage

### Command-line tools

After installation, the following convenience commands are available:

- `mace-jax-train`
- `mace-jax-preprocess`
- `mace-jax-predict`
- `mace-jax-train-plot`
- `mace-jax-from-torch`
- `mace-jax-hdf5-benchmark`
- `mace-jax-hdf5-info`
- `mace-create-lammps-model`

#### `mace-jax-train`

Runs training driven by gin-config files. Example:

```sh
mace-jax-train configs/aspirin_small.gin \
  --seed 0 \
  --print-config
```

The operative configuration is saved alongside the run logs.

A repository helper mirrors this example and runs a one-interval smoke test on
the bundled 3BPA dataset:

```sh
python scripts/run_aspirin_smoke_test.py --print-config
```

The script exposes knobs for the dataset paths, seed, device, log directory, and
epoch length, making it convenient for CI or quick local validation without
editing gin files.

Additional convenience flags let you adjust common gin settings directly from the CLI:

- `--torch-checkpoint PATH`: import parameters from a Torch checkpoint (converted on the fly via `mace-jax-from-torch` utilities).
- `--torch-head NAME`: select a specific head from the imported Torch model.
- `--torch-param-dtype {float32,float64}`: override the dtype used for imported parameters.
- `--train-path/--valid-path/--test-path`: point datasets to new files without editing the gin config.
- `--statistics-file PATH`: reuse `statistics.json` from preprocessing (sets atomic numbers,
  E0s, scaling mean/std, and average neighbor count).
- `--r-max VALUE`: synchronise the cutoff used in both dataset construction and model definition.
  For streaming datasets, `--batch-max-edges` (or `n_edge` in gin) sets the edge cap.

##### cuequivariance (cueq) / conv_fusion

MACE-JAX exposes cuequivariance with CLI flags that mirror the Torch interface:

```sh
# Enable cueq + conv_fusion (similar to mace --enable_cueq).
mace-jax-train configs/aspirin_small.gin --enable_cueq

# cueq-only style (similar to mace --only_cueq): force cueq backend + optimizations.
mace-jax-train configs/aspirin_small.gin --only_cueq
```

Notes:
- `conv_fusion` is enabled automatically when CUDA is detected.
- You can still override `CuEquivarianceConfig` via `--binding` if you need custom
  layout/group settings.
 - `--cueq-optimize-all/--no-cueq-optimize-all` and
   `--cueq-conv-fusion/--no-cueq-conv-fusion` provide direct CLI control over
   these settings.
 - `--cueq-layout {mul_ir,ir_mul}` and `--cueq-group {O3,O3_e3nn}` mirror the
   Torch defaults (use `ir_mul`/`O3_e3nn` for `--only_cueq`).

For instance, fine-tuning a Torch foundation model against a new dataset can be done with:

```sh
mace-jax-train configs/finetune.gin \
  --torch-checkpoint checkpoints/foundation.pt \
  --torch-head Surface \
  --train-path data/surface.xyz \
  --valid-path None \
  --r-max 4.5 \
  --print-config
```

##### Optional logging & averaging

- **Weights & Biases** logging is enabled via CLI flags: `--wandb` launches a
  run with the automatically generated tag/operative gin, and you can supply
  `--wandb-project`, `--wandb-entity`, `--wandb-tag`, etc. No manual gin edits
  are necessary. Metrics for each evaluation split plus per-interval timing are
  streamed to the configured run.

- **Stochastic Weight Averaging (SWA)** is available through `--swa` (with
  `--swa-start`, `--swa-every`, `--swa-min-snapshots`, … mirroring the Torch
  CLI). Once the requested number of snapshots has been accumulated,
  evaluations use the SWA parameters while the raw/EMA weights continue to be
  optimised.

- **Gradient clipping & EMA** can be toggled directly from the CLI using
  `--clip-grad VALUE` and `--ema-decay VALUE`. These map to the same behaviour
  as the Torch `--clip_grad`/`--ema` flags and remove the need for explicit gin
  bindings.
- **Optimizer & scheduler** settings can be specified via `--optimizer`
  (`adam`, `adamw`, `amsgrad`, `sgd`, `schedulefree`), `--lr`, `--weight-decay`,
  `--schedule-free-b1`, `--schedule-free-weight-lr-power`, `--scheduler`
  (`constant`, `exponential`, `piecewise_constant`, `plateau`), and
  `--lr_scheduler_gamma`.
  These bind directly into the gin optimizer helper, mirroring the Torch CLI.
- **Foundation models** can be pulled directly via the same interface as Torch
  MACE. Use `--foundation_model small` (or `medium`, `large`, `small_off`, …) to
  download and initialize from the released checkpoints, or pass a custom path.
  The CLI adjusts the cutoff, learning-rate defaults, and multihead finetuning
  knobs just like `mace run_train`.
- **Multihead finetuning** mirrors the Torch defaults via `--multiheads_finetuning`
  (optionally with `--force_mh_ft_lr`): learning rate/EMA defaults are adjusted
  automatically when combined with `--foundation_model`, so migrating scripts
  can keep the same behaviour.
- **Distributed training** can run multi-process on a single node or across hosts.
  For single-node multi-GPU runs, the default `--launcher auto` will start one process
  per visible GPU and enable `jax.distributed` automatically when more than one GPU is
  visible. Use `CUDA_VISIBLE_DEVICES` to restrict which GPUs are used, or pass
  `--launcher none` to force single-process mode.

  ```sh
  # Single node, two GPUs (auto spawns 2 processes).
  CUDA_VISIBLE_DEVICES=0,1 \
  mace-jax-train configs/aspirin_small.gin \
    --device cuda \
    --launcher auto
  ```

  For multi-host launches you still need to provide `--distributed` plus the
  process topology (`--process-count`, `--process-index`, and optional
  `--coordinator-address/--coordinator-port`). When distributed is enabled the CLI
  initialises `jax.distributed`, shards the training/validation/test datasets per
  process deterministically, and only writes logs/checkpoints from rank 0.
  Environment variables such as `JAX_PROCESS_COUNT`, `JAX_PROCESS_INDEX` or
  Slurm launch variables can also be used; they override the CLI defaults automatically.
A typical 2-host launch (one process per host) looks like:

  ```sh
  JAX_PROCESS_COUNT=2 \
  JAX_PROCESS_INDEX=${SLURM_PROCID} \
  mace-jax-train configs/aspirin_small.gin \
    --device cuda \
    --distributed \
    --coordinator-address host0 \
    --coordinator-port 12345
  ```

#### `mace-jax-predict`

Runs inference on XYZ or HDF5 datasets and writes `.npz` outputs (one per input).
For streaming HDF5 inputs, predictions are ordered by the stable `graph_id` and
written as NumPy arrays:

```sh
# XYZ inference.
mace-jax-predict checkpoints/model.ckpt data/structures.xyz

# HDF5 inference (uses cached streaming stats if present).
mace-jax-predict checkpoints/model.ckpt data/hdf5/train_0.h5 \
  --batch-max-edges 6000 \
  --num-workers 8
```

Notes:
- `--batch-max-edges` is required for streaming HDF5 predictions unless cached
  streaming stats are present next to the file.
- `--batch-max-nodes` is optional for predictions if you want to override the
  cached node cap.

#### `mace-jax-preprocess`

Converts one or more XYZ files into MACE-style streaming HDF5 files and (optionally)
computes dataset statistics. The outputs are written to `<h5_prefix>/train`,
`<h5_prefix>/val`, and `<h5_prefix>/test`, plus `statistics.json` when requested.

```sh
mace-jax-preprocess \
  --train_file data/train.xyz \
  --valid_file data/valid.xyz \
  --h5_prefix data/hdf5/ \
  --r_max 5.0 \
  --compute_statistics \
  --atomic_numbers "[1, 6, 8]" \
  --E0s "average"
```

Pass the resulting `statistics.json` to training via `mace-jax-train --statistics-file`
to reuse the computed scaling and average neighbor counts.

#### `mace-jax-train-plot`

Produces loss/metric curves from `.metrics` logs generated during training (JSON-lines logs in the run directory):

```sh
mace-jax-train-plot --path results --keys rmse_e_per_atom,rmse_f --output-format pdf
```

The command accepts either a directory (all `.metrics` files are processed) or a single metrics file. It plots
`eval_valid` and `eval_train` loss curves (with per-head suffixes like `eval_valid:Surface` when present), plus the
extra metric keys requested via `--keys`.

#### `mace-jax-hdf5-info`

Prints a quick summary of MACE-style HDF5 datasets (batch counts, graph counts,
property keys, etc.). If a streaming stats cache is present, it also reports the
cached padding caps:

```sh
mace-jax-hdf5-info data/hdf5/train_*.h5
```

#### `mace-jax-hdf5-benchmark`

Measures HDF5 read throughput for MACE-style datasets (sequential and optional
shuffled access patterns):

```sh
mace-jax-hdf5-benchmark data/hdf5/train_0.h5 --count 1000 --repeat 3 --shuffle
```

#### `mace-create-lammps-model`

Converts a Torch MACE checkpoint to the JAX MLIAP format:

```sh
mace-create-lammps-model checkpoints/model.pt --dtype float32 --output exported_model.pkl
```

Add `--head NAME` when exporting a multi-head model.

#### `mace-jax-from-torch`

Performs Torch→JAX parameter conversion and (optionally) prediction. It can also
download and import pre-trained foundation models when `--torch-model` is not
provided. To compute energies for the provided test structure:

```sh
mace-jax-from-torch --torch-model checkpoints/model.pt --predict tests/test_data/simple.xyz
```

If `--output` is omitted the converted parameters are written to `<checkpoint>-jax.npz`.

You can try this with one of the released foundation models (downloaded automatically):

```sh
mace-jax-from-torch --foundation mp --model-name small --predict tests/test_data/simple.xyz
```

All commands can be invoked via `python -m mace_jax.<module>` if preferred.

### Expert usage (internal APIs)

If you want to call the CLI helpers directly from Python (for notebooks or
custom pipelines), the internal entrypoints are exposed in the CLI modules.
These helpers are not part of the stable public API (they use leading
underscores), so prefer the CLI for long-term scripts.

#### Predictions (`_predict_xyz` / `_predict_hdf5`)

Both helpers live in `mace_jax.cli.mace_jax_predict` and return
`(graph_ids, outputs)`. For HDF5 inputs the outputs are **sorted by graph_id**
within the local process, so they can be merged across processes later if
needed.

```py
from pathlib import Path

from mace_jax.cli import mace_jax_predict
from mace_jax.tools import bundle as bundle_tools

bundle = bundle_tools.load_model_bundle("checkpoints/model-jax.npz", "")
head_name, head_to_index = mace_jax_predict._resolve_head_mapping(
    bundle.config, head=None
)
predictor = mace_jax_predict._build_predictor(
    bundle.module, compute_forces=True, compute_stress=True
)

# XYZ prediction (equivalent to CLI on .xyz input).
graph_ids, outputs = mace_jax_predict._predict_xyz(
    Path("data/structures.xyz"),
    predictor=predictor,
    params=bundle.params,
    model_config=bundle.config,
    head_name=head_name,
    head_to_index=head_to_index,
)

# Streaming HDF5 prediction (requires batch_max_edges).
graph_ids, outputs = mace_jax_predict._predict_hdf5(
    Path("data/train_0.h5"),
    predictor=predictor,
    params=bundle.params,
    model_config=bundle.config,
    head_name=head_name,
    head_to_index=head_to_index,
    batch_max_edges=6000,
    batch_max_nodes=None,
    prefetch_batches=2,
    num_workers=8,
    progress_bar=True,
)
```

#### Training (`run_training`)

The CLI entrypoint in `mace_jax.cli.mace_jax_train` is also reusable in-process
as long as gin is configured first:

```py
import gin

from mace_jax.cli import mace_jax_train

gin.parse_config_files_and_bindings(
    ["configs/aspirin_small.gin"],
    bindings=[],
)
mace_jax_train.run_training()
```

For even finer control you can call the same building blocks used by the CLI:
`mace_jax.tools.gin_datasets.datasets`, `mace_jax.tools.gin_model.model`,
`mace_jax.tools.gin_functions.loss/optimizer/train`, and the helper functions
inside `mace_jax.cli.mace_jax_train` (such as `_build_predictor`).

### Streaming HDF5 loader (mace-jax specific)

Fixed-shape batches let JAX/XLA compile once and reuse the same executable, so
the streaming loader is designed to keep batch shapes stable across epochs.
MACE‑JAX compiles the model with **fixed batch shapes**, so it cannot accept
variable‑sized batches on the fly. To make this efficient, the streaming loader
runs a stats scan to derive fixed padding caps and then builds batches on the fly
within those caps. This avoids repeated recompilations and keeps throughput stable
on large HDF5 datasets.

The only sizing knob you must provide is `n_edge` (also exposed as `--batch-max-edges`
or `n_edge` in gin). `n_node` is **not** supported for streaming datasets; it is
derived automatically from the streaming stats scan (with an optional percentile
cap for tighter node padding).

How `n_edge` is used:

- **Edge cap / graph feasibility:** Each graph’s edge count is checked against the
  configured `n_edge` limit. If a graph exceeds the limit, the loader raises the cap
  and logs a warning so all graphs fit.
- **Batch packing (knapsack‑like):** The stats pass uses greedy packing to estimate
  typical batch sizes under the `n_edge` cap.
- **Derived node/graph caps:** The streaming stats compute node/graph caps from the
  edge-limited batches. `n_node` defaults to the 90th percentile of batch node
  totals, but is never lower than the largest single-graph node count. `n_graph`
  uses the maximum observed graph count per batch. These become the fixed
  `n_node`/`n_graph` pads used for JAX compilation.
- **On‑the‑fly batching:** During training/eval, worker processes read HDF5 entries
  directly, pack batches until a cap is hit, pad, and emit the batch. There are
  no separate reader processes. Shuffling is optional and, when enabled, uses a
  deterministic per-epoch shuffle of global graph indices.
- **Graph IDs:** Each graph is tagged with a stable global `graph_id` so streaming
  prediction outputs can be reordered to the original HDF5 order when needed.
- **Multi-process sharding:** For distributed runs, global graph indices are
  sharded round-robin by `process_index` across processes; each process reads only
  its own slice of indices but can span all HDF5 files.
- **Persisted caps:** The derived `n_nodes/n_edges/n_graphs` caps are stored in the
  streaming stats. Train/valid/test each get their own stats so loaders can be reused
  independently.

In short, `n_edge` controls memory and compilation shape: it is the fixed edge budget
per batch. The loader packs graphs to stay under this budget, computes the node/graph
padding needed, and then reuses these fixed shapes for every epoch.

Streaming datasets accept a single HDF5 file, a directory of `.h5/.hdf5` files, or a
glob pattern. The loader expands all matching files and derives padding caps across
the combined dataset.

Streaming loader knobs (CLI or gin):
- `--batch-max-edges` / `n_edge`: edge cap for fixed-shape batches.
- `--suffle`: enable/disable per-epoch shuffling for training (default: true).
  Validation/test loaders never shuffle.
- `--batch-node-precentile`: percentile for node padding cap (default: 90);
  caps never drop below the largest single graph.
- `--stream-train-max-batches`: optional cap on batches per epoch.

### Configuration

Links to the files containing the functions configured by the gin config file.

- [`flags`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`logs`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`datasets`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_datasets.py)
- [`model`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_model.py) and [here](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/models.py)
- [`loss`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/loss.py)
    if `loss.energy_weight`, `loss.forces_weight` or `loss.stress_weight` is nonzero the loss will be applied
- [`optimizer`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`train`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)

## Contributions

We are happy to accept pull requests under an [MIT license](https://choosealicense.com/licenses/mit/). Please copy/paste the license text as a comment into your pull request.

## References

If you use this code, please cite our papers:
```text
@misc{Batatia2022MACE,
  title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2206.07697},
  eprint = {2206.07697},
  eprinttype = {arxiv},
  doi = {10.48550/ARXIV.2206.07697},
  archiveprefix = {arXiv}
}
@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr or geiger.mario@gmail.com.

## License

MACE is published and distributed under the [MIT](MIT.md).
