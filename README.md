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

### Training

To train a MACE model, you can use the `run_train.py` script:

```sh
python -m mace_jax.run_train config.gin
```

An example of configuration file is located in the directory `configs`.

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
