# MACE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :rocket: [JAX](https://github.com/google/jax)

This repository contains a **porting** of MACE in **jax** developed by
**Mario Geiger** and Ilyes Batatia.

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
python setup.py develop
```

## Usage

### Command-line tools

After installation, the following convenience commands are available:

#### `mace-jax-train`

Runs training driven by gin-config files. Example:

```sh
mace-jax-train configs/aspirin_small.gin \
  -b "mace_jax.tools.gin_functions.flags.seed=0" \
  --print-config
```

Use `--dry-run` to validate the configuration without launching training. The operative configuration is saved alongside the run logs.

#### `mace-jax-train-plot`

Produces loss/metric curves from `.metrics` logs generated during training:

```sh
mace-jax-train-plot --path results --keys rmse_e_per_atom,rmse_f --output-format pdf
```

The command accepts either a directory (all `.metrics` files are processed) or a single metrics file.

#### `mace-create-lammps-model`

Converts a Torch MACE checkpoint to the JAX MLIAP format:

```sh
mace-create-lammps-model checkpoints/model.pt --dtype float32 --output exported_model.pkl
```

Add `--head NAME` when exporting a multi-head model.

#### `mace-torch2jax`

Performs Torch→JAX parameter conversion and (optionally) prediction. To compute energies for the provided test structure:

```sh
mace-torch2jax checkpoints/model.pt --predict tests/test_data/simple.xyz
```

If `--output` is omitted the converted parameters are written to `<checkpoint>-jax.npz`.

You can try this with one of the released foundation models (downloaded automatically):

```sh
mace-torch2jax --foundation mp --model-name small --predict tests/test_data/simple.xyz
```

All commands can be invoked via `python -m mace_jax.<module>` if preferred.

### Configuration

Links to the files containing the functions configured by the gin config file.

- [`flags`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`logs`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`datasets`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/tools/gin_datasets.py)
- [`model`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/tools/gin_model.py) and [here](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/modules/models.py)
- [`loss`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/modules/loss.py)
    if `loss.energy_weight`, `loss.forces_weight` or `loss.stress_weight` is nonzero the loss will be applied
- [`optimizer`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`train`](https://github.com/ilyes319/mace-jax/blob/main/mace_jax/tools/gin_functions.py)

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
