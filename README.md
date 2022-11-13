# MACE

This repository contains a **porting** of MACE in **jax** developed by
Mario Geiger and Ilyes Batatia.

## Installation

Requirements: TODO

### conda installation

TODO

### pip installation

TODO

## Usage

### Training

To train a MACE model, you can use the `run_train.py` script:

```sh
python ./mace-jax/scripts/run_train.py ./mace-jax/scripts/config.gin
```

To control the model's size, you need to change `model.hidden_irreps`. For most applications, the recommended default model size is `model.hidden_irreps='256x0e'` (meaning 256 invariant messages) or `model.hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`.

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above. To label them in the training set, set `config_type=IsolatedAtom` in their info fields. If you prefer not to use or do not know the energies of the isolated atoms, you can use the option `model.atomic_energies='average'` which estimates the atomic energies using least squares regression.

The precision can be changed using the keyword ``flags.dtype``, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).

The keywords ``datasets.batch_size`` and ``optimizer.max_num_epochs`` should be adapted based on the size of the training set. The batch size should be increased when the number of training data increases, and the number of epochs should be decreased. An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

### Evaluation

TODO

## Development

We use `black`, `isort`, `pylint`, and `mypy`.
Run the following to format and check your code:
```sh
bash ./scripts/run_checks.sh
```

We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.

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

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr.

## License

MACE is published and distributed under the [Academic Software License v1.0 ](ASL.md).
