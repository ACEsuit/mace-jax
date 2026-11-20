"""Magnetic MACE with differentiable positional and moment dependence."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from e3nn_jax import Irreps, spherical_harmonics
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.modules.magnetic_blocks import EquivariantProductBasisWithSelfMagmomBlock
from mace_jax.modules.models import ScaleShiftMACE
from mace_jax.modules.utils import add_output_interface
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.dtype import default_dtype


def chebyshev_basis(x, size, include_constant=False):
    values = [jnp.ones_like(x), x]
    start = 0 if include_constant else 1
    for _ in range(2, size + start):
        values.append(2 * x * values[-1] - values[-2])
    return jnp.stack(values[start : start + size], axis=-1)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class MagneticScaleShiftMACE(ScaleShiftMACE):
    magnetic = True

    def __init__(
        self,
        *,
        m_max,
        num_mag_radial_basis,
        max_m_ell,
        use_magmom_one_body=False,
        num_mag_radial_basis_one_body=0,
        rngs,
        **kwargs,
    ):
        self.num_mag_radial_basis = int(num_mag_radial_basis)
        self.max_m_ell = int(max_m_ell)
        self.use_magmom_one_body = use_magmom_one_body
        self.num_mag_radial_basis_one_body = int(num_mag_radial_basis_one_body)
        if self.num_mag_radial_basis < 1 or self.max_m_ell < 0:
            raise ValueError(
                'Magnetic basis size must be positive and max_m_ell nonnegative.'
            )
        if use_magmom_one_body and self.num_mag_radial_basis_one_body < 1:
            raise ValueError(
                'Magnetic one-body terms require at least one basis function.'
            )
        if kwargs.get('use_last_readout_only') or kwargs.get('embedding_specs'):
            raise ValueError(
                'Magnetic MACE does not support last-readout-only or feature embeddings.'
            )
        kwargs['keep_last_layer_irreps'] = False
        # Upstream magnetic models always apply the radial cutoff internally.
        kwargs['apply_cutoff'] = True
        super().__init__(rngs=rngs, **kwargs)
        maximum = np.asarray(m_max)
        if maximum.shape != (self.num_elements,) or not np.all(
            np.isfinite(maximum) & (maximum > 0)
        ):
            raise ValueError(
                'm_max must contain one finite positive value per element.'
            )
        self.m_max = ConfigVar(jnp.asarray(maximum, dtype=default_dtype()))
        if use_magmom_one_body:
            shape = (
                self.num_elements,
                self.num_mag_radial_basis_one_body,
                len(self._heads),
            )
            self.onebody_magmombasis_coeffs = nnx.Param(
                jax.random.normal(rngs.params(), shape, dtype=default_dtype())
            )
            self.one_body_magmom_const_correction = ConfigVar(
                jnp.zeros((self.num_elements, len(self._heads)), dtype=default_dtype())
            )

    def _magnetic_irreps(self):
        return dict(
            magmom_node_inv_feats_irreps=Irreps(f'{self.num_mag_radial_basis}x0e'),
            magmom_node_attrs_irreps=Irreps.spherical_harmonics(self.max_m_ell),
        )

    def _make_interaction(self, cls, **kwargs):
        return cls(**self._magnetic_irreps(), **kwargs)

    def _make_product(self, **kwargs):
        # Match the actual Torch product construction, which does not pass the
        # model-level reduced-CG/element-agnostic flags to magnetic products.
        kwargs.update(use_reduced_cg=False, use_agnostic_product=False)
        return EquivariantProductBasisWithSelfMagmomBlock(
            **self._magnetic_irreps(), **kwargs
        )

    def _scaled_moments(self, data):
        maximum = jnp.asarray(self.m_max[...])[jnp.argmax(data['node_attrs'], axis=-1)]
        # Squared norms avoid the undefined derivative of norm at zero.
        squared = jnp.sum(data['magmom'] ** 2, axis=-1) / maximum**2
        return 1 - 2 * jnp.minimum(squared, 1)

    def _interaction_inputs(self, data):
        moments = data['magmom']
        if moments.shape != data['positions'].shape:
            raise ValueError('magmom must have shape [number of atoms, 3].')
        return dict(
            magmom_node_inv_feats=chebyshev_basis(
                self._scaled_moments(data), self.num_mag_radial_basis
            ),
            magmom_node_attrs=spherical_harmonics(
                Irreps.spherical_harmonics(self.max_m_ell),
                moments,
                normalize=False,
                normalization='integral',
            ).array,
        )

    def _node_energy_correction(self, data):
        if not self.use_magmom_one_body:
            return 0.0
        basis = chebyshev_basis(
            self._scaled_moments(data), self.num_mag_radial_basis_one_body, True
        )
        coeffs = jnp.einsum(
            'ns,sbh->nbh', data['node_attrs'], self.onebody_magmombasis_coeffs[...]
        )
        correction = data['node_attrs'] @ self.one_body_magmom_const_correction[...]
        energy = jnp.einsum('nb,nbh->nh', basis, coeffs) - correction
        heads = data.get('head', jnp.zeros(data['ptr'].shape[0] - 1, dtype=int))[
            data['batch']
        ]
        return energy[jnp.arange(len(heads)), heads]

    def __call__(self, data, *, compute_node_feats=True, **kwargs):
        if (
            kwargs.get('lammps_mliap')
            or kwargs.get('lammps_class') is not None
            or data.get('lammps_class') is not None
            or kwargs.get('comm') is not None
        ):
            raise ValueError('Magnetic MACE does not yet support ghost exchange.')
        return ScaleShiftMACE._energy_fn(
            self, data, compute_node_feats=compute_node_feats, **kwargs
        )


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MagneticSCFMACE(nnx.Module):
    """Relax magnetic moments using L-BFGS before evaluating the model.

    Inputs supply the initial moments explicitly. Returned moments can be reused
    as the next initial guess. The solver is stateless and supports padded
    batches and JIT. Energy histories have a fixed length, with the last value
    repeated after convergence; ``scf_steps`` counts accepted update iterations.
    Derivatives hold the relaxed moments fixed, matching Torch's SCF convention.
    """

    def __init__(
        self,
        model=None,
        *,
        n_scf_step=10,
        scf_tol=1e-5,
        scf_logging=False,
        scf_step_size=1.0,
        use_scf=True,
        use_collinear=False,
        rngs=None,
        **kwargs,
    ):
        self.magmom_mace = (
            model if model is not None else MagneticScaleShiftMACE(rngs=rngs, **kwargs)
        )
        self.n_scf_step = int(n_scf_step)
        self.scf_tol = float(scf_tol)
        self.scf_logging = bool(scf_logging)
        self.scf_step_size = float(scf_step_size)
        self.use_scf = bool(use_scf)
        self.use_collinear = bool(use_collinear)
        if self.n_scf_step < 0 or self.scf_tol < 0 or self.scf_step_size <= 0:
            raise ValueError(
                'SCF steps/tolerance must be nonnegative and step size positive.'
            )

    def __getattr__(self, name):
        inner = self.__dict__.get('magmom_mace')
        if inner is None or name == 'magmom_mace':
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, data, **flags):
        if 'magmom' not in data:
            raise ValueError('Magnetic SCF requires explicit initial magmom values.')
        initial = jax.lax.stop_gradient(data['magmom'])
        params = initial[:, 2] if self.use_collinear else initial
        compose = lambda values: (
            initial.at[:, 2].set(values) if self.use_collinear else values
        )
        objective = lambda values: self.magmom_mace._energy_fn(
            dict(data, magmom=compose(values))
        )['energy'].sum()
        value_grad = jax.value_and_grad(objective)
        value, grad = value_grad(params)
        steps = jnp.array(0, dtype=jnp.int32)
        if self.use_scf and self.n_scf_step:
            solver = optax.lbfgs(learning_rate=self.scf_step_size)
            state = solver.init(params)

            def step(carry, _):
                params, state, value, grad, steps = carry
                active = jnp.max(jnp.abs(grad)) > self.scf_tol

                def update():
                    updates, next_state = solver.update(
                        grad, state, params, value=value, grad=grad, value_fn=objective
                    )
                    next_params = optax.apply_updates(params, updates)
                    next_value, next_grad = value_grad(next_params)
                    if self.scf_logging:
                        jax.debug.print(
                            'Magnetic SCF energy: {value}', value=next_value
                        )
                    return next_params, next_state, next_value, next_grad, steps + 1

                carry = jax.lax.cond(active, update, lambda: carry)
                return carry, carry[2]

            (params, _, final_value, grad, steps), history = jax.lax.scan(
                step, (params, state, value, grad, steps), None, length=self.n_scf_step
            )
            history = jnp.concatenate((value[None], history))
        else:
            history = value[None]
        moments = jax.lax.stop_gradient(compose(params))
        result = self.magmom_mace(dict(data, magmom=moments), **flags)
        result.update(
            equilibrated_magmom=moments,
            scf_energy_history=history,
            scf_steps=steps,
            scf_converged=jnp.max(jnp.abs(grad)) <= self.scf_tol,
        )
        return result
