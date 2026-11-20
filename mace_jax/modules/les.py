"""MACE with native JAX latent electrostatics (MACELES)."""

import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps, reduced_tensor_product_basis
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace_jax.modules.electrostatics import les_electrostatics
from mace_jax.modules.les_blocks import LinearLesReadoutBlock, NonLinearLesReadoutBlock
from mace_jax.modules.les_data import ATOMIC_POLARIZABILITIES, TYPICAL_CHARGES
from mace_jax.modules.models import ScaleShiftMACE
from mace_jax.modules.utils import add_output_interface
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.scatter import scatter_sum


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class MACELES(ScaleShiftMACE):
    """LES charges, multipoles, and induced response on the MACE backbone.

    ``node_energy`` retains the local contribution, as in Torch MACE;
    ``les_energy`` supplies the additional graph-level electrostatic energy.
    ``reciprocal_max_index`` bounds the static reciprocal grid and is checked
    against every periodic cell at runtime (including compiled calls).
    """

    cell_dependent = True

    def __init__(self, *, les_arguments=None, rngs, **kwargs):
        args = dict(les_arguments or {})
        self.les_arguments = args
        for key in ('sigma', 'dl', 'epsilon_factor'):
            if not np.isfinite(args.get(key, 1.0)) or args.get(key, 1.0) <= 0:
                raise ValueError(f'LES {key} must be finite and positive.')
        kwargs['keep_last_layer_irreps'] = True
        super().__init__(rngs=rngs, **kwargs)
        self.compute_bec = bool(args.get('compute_bec', False))
        self.bec_output_index = args.get('bec_output_index')
        if self.bec_output_index not in (None, 0, 1, 2):
            raise ValueError('bec_output_index must be None, 0, 1, or 2.')
        self.use_dipoles = args.get('use_dipole', False)
        self.use_quads = args.get('use_quad', False)
        self.use_induced_charges = args.get('use_induced_charge', False)
        self.use_induced_dipoles = args.get('use_induced_dipole', False)
        self.use_anisotropic_polarizability = args.get(
            'use_anisotropic_polarizability', False
        )
        self.alpha_irreps = args.get('alpha_irreps', '0e+1o+2e')
        self.les_output_scale = args.get('output_scale', 0.1)
        self.les_kappa_scale = args.get('kappa_scale', 0.01)
        self.les_alpha_scale = args.get('alpha_scale', 0.01)
        basis = reduced_tensor_product_basis('ij=ji', i=Irreps('1o')).array.transpose(
            2, 0, 1
        )
        self.change_of_basis = ConfigVar(jnp.asarray(basis, dtype=default_dtype()))
        self.change_of_basis_quads = ConfigVar(
            jnp.asarray(basis[1:], dtype=default_dtype())
        )
        layers = (
            [self.num_interactions - 1]
            if self.use_last_readout_only
            else range(self.num_interactions)
        )
        self.les_readouts = nnx.List(
            [self._make_readout(i, self._hidden_irreps, rngs) for i in layers]
        )
        self.les_kappa_readouts = nnx.List(
            [self._make_readout(i, self._hidden_irreps, rngs) for i in layers]
            if self.use_induced_charges
            else []
        )
        self.les_alpha_readouts = nnx.List(
            [self._make_readout(i, self._hidden_irreps, rngs) for i in layers]
            if self.use_induced_dipoles
            and (not self.use_anisotropic_polarizability or '0e' in self.alpha_irreps)
            else []
        )

        def linear(output, enabled):
            if isinstance(self.readouts[0], LinearReadoutBlock):
                make = lambda: LinearReadoutBlock(
                    self._hidden_irreps,
                    Irreps(output),
                    self.equivariance_config,
                    rngs=rngs,
                )
            else:
                make = lambda: NonLinearReadoutBlock(
                    self._hidden_irreps,
                    (len(self._heads) * self._mlp_irreps).simplify(),
                    self.gate,
                    Irreps(output),
                    len(self._heads),
                    self.equivariance_config,
                    rngs=rngs,
                )
            return nnx.List([make() for _ in layers] if enabled else [])

        def dyadic(enabled, positive):
            readout = (
                NonLinearLesReadoutBlock
                if args.get('alpha_1o_nonlinear_readout', False)
                else LinearLesReadoutBlock
            )
            options = (
                {} if readout is NonLinearLesReadoutBlock else {'make_w_pos': positive}
            )
            return nnx.List(
                [
                    readout(
                        self._hidden_irreps,
                        equivariance_config=self.equivariance_config,
                        rngs=rngs,
                        **options,
                    )
                    for _ in layers
                ]
                if enabled
                else []
            )

        has_1o, has_2e = (
            self._hidden_irreps.count('1o'),
            self._hidden_irreps.count('2e'),
        )
        anisotropic = self.use_induced_dipoles and self.use_anisotropic_polarizability
        if anisotropic and not (has_1o or has_2e):
            raise ValueError('Anisotropic polarizability requires 1o or 2e channels.')
        self.les_u_readouts = linear('1o', self.use_dipoles)
        self.les_quad_2e_readouts = linear('2e', self.use_quads and has_2e)
        self.les_quad_1o_readouts = dyadic(self.use_quads and has_1o, False)
        self.les_alpha_2e_readouts = linear(
            '0e+2e', anisotropic and has_2e and '2e' in self.alpha_irreps
        )
        self.les_alpha_1o_readouts = dyadic(
            anisotropic and has_1o and '1o' in self.alpha_irreps,
            args.get('alpha_1o_linear_w_pos', True),
        )

        def baseline(flag, key, table, factor):
            values = args.get(key)
            if values is None:
                if args.get(flag, False):
                    if any(z < 1 or z >= len(table) for z in self.atomic_numbers):
                        raise ValueError(
                            f'No default LES baseline for these elements; supply {key}.'
                        )
                    values = [table[z] * factor for z in self.atomic_numbers]
                else:
                    values = [0.0] * self.num_elements
            values = np.asarray(values)
            if values.shape != (self.num_elements,) or not np.isfinite(values).all():
                raise ValueError(f'{key} must contain one finite value per element.')
            return ConfigVar(jnp.asarray(values, dtype=default_dtype()))

        self.fixed_charge_values = baseline(
            'use_fixed_atomic_charges',
            'fixed_charge_values',
            TYPICAL_CHARGES,
            args.get('fixed_atomic_charges_scaling_factor', 0.5),
        )
        self.atomic_alpha_values = baseline(
            'use_atomic_alpha',
            'atomic_alpha_values',
            ATOMIC_POLARIZABILITIES,
            0.1481847 / 14.3996,
        )

    def __call__(
        self,
        data,
        *,
        compute_node_feats=True,
        compute_bec=False,
        _skip_bec=False,
        **kwargs,
    ):
        if (
            kwargs.get('lammps_mliap')
            or kwargs.get('lammps_class') is not None
            or data.get('lammps_class') is not None
            or kwargs.get('comm') is not None
        ):
            raise ValueError('MACELES requires complete graphs for electrostatics.')
        output = ScaleShiftMACE._energy_fn(
            self, data, compute_node_feats=True, **kwargs
        )
        feats = jnp.split(output['node_feats'], self.num_interactions, axis=-1)
        if len(self.readouts) == 1:
            feats = feats[-1:]
        batch = data['batch']
        n_graphs = data['ptr'].shape[0] - 1
        heads = data.get('head', jnp.zeros(n_graphs, dtype=int))[batch]
        rows = jnp.arange(len(batch))
        mask = jnp.asarray(
            data.get('node_mask', jnp.ones(len(batch))), dtype=data['positions'].dtype
        )

        def read(blocks, scalar=False, basis=None):
            if not blocks:
                return None
            values = [block(x, heads) for block, x in zip(blocks, feats)]
            value = sum(values)
            if scalar:
                value = value[rows, heads]
            if basis is not None:
                value = jnp.einsum('na,aij->nij', value, basis)
            return value * mask.reshape((-1,) + (1,) * (value.ndim - 1))

        q = read(self.les_readouts, scalar=True) * self.les_output_scale
        u = read(self.les_u_readouts)
        u = None if u is None else u * self.les_output_scale
        quad_values = [
            read(self.les_quad_1o_readouts),
            read(self.les_quad_2e_readouts, basis=self.change_of_basis_quads[...]),
        ]
        quad_values = [v for v in quad_values if v is not None]
        quad = sum(quad_values) * self.les_output_scale if quad_values else None
        if quad is not None:
            quad -= jnp.trace(quad, axis1=-2, axis2=-1)[:, None, None] * jnp.eye(3) / 3
        kappa = read(self.les_kappa_readouts, scalar=True)
        if kappa is not None:
            kappa *= self.les_kappa_scale
            if self.les_arguments.get('make_kappa_positive', False):
                kappa = kappa**2
        alpha = read(self.les_alpha_readouts, scalar=True)
        if self.use_anisotropic_polarizability:
            alpha = None if alpha is None else alpha[:, None, None] * jnp.eye(3)
            values = [
                alpha,
                read(self.les_alpha_1o_readouts),
                read(self.les_alpha_2e_readouts, basis=self.change_of_basis[...]),
            ]
            values = [v for v in values if v is not None]
            alpha = sum(values) if values else None
        if alpha is not None:
            alpha *= self.les_alpha_scale
            if self.les_arguments.get('make_alpha_positive', False):
                alpha = (
                    alpha**2
                    if alpha.ndim == 1
                    else jnp.einsum('nij,nkj->nik', alpha, alpha)
                )
        if self.les_arguments.get('use_fixed_atomic_charges', False):
            q += data['node_attrs'] @ self.fixed_charge_values[...] * mask
        if alpha is not None and self.les_arguments.get('use_atomic_alpha', False):
            baseline = data['node_attrs'] @ self.atomic_alpha_values[...] * mask
            alpha += (
                baseline if alpha.ndim == 1 else baseline[:, None, None] * jnp.eye(3)
            )
        options = {
            key: self.les_arguments[key]
            for key in (
                'sigma',
                'dl',
                'remove_self_interaction',
                'reciprocal_max_index',
            )
            if key in self.les_arguments
        }
        les_data = dict(data)
        if 'e_ext' in data:
            les_data['external_field'] = data['e_ext']
        energy, charges, dipoles = les_electrostatics(
            les_data, q, u, quad, kappa, alpha, **options
        )
        output.update(
            energy=output['energy'] + energy,
            les_energy=energy,
            latent_charges=charges,
            latent_dipoles=dipoles if u is not None or alpha is not None else None,
            latent_quads=quad,
            latent_kappas=kappa,
            latent_alphas=alpha,
            BEC=None,
        )
        if not compute_node_feats:
            output['node_feats'] = None
        if not _skip_bec and (compute_bec or self.compute_bec):
            output['BEC'] = self._born_charges(data)
        return output

    def _born_charges(self, data):
        batch = data['batch']
        n_graphs = data['ptr'].shape[0] - 1
        cell = data['cell'].reshape(n_graphs, 3, 3)
        periodic = jnp.any(
            data.get('pbc', jnp.ones((n_graphs, 3), dtype=bool)), axis=-1
        ) & (jnp.linalg.det(cell) >= 1e-6)
        cell = jnp.where(periodic[:, None, None], cell, jnp.eye(3))
        mask = jnp.asarray(
            data.get('node_mask', jnp.ones(len(batch))), dtype=data['positions'].dtype
        )

        def moments(pos):
            # Avoid recursively computing BEC when it is enabled in the config.
            return MACELES._energy_fn(self, dict(data, positions=pos), _skip_bec=True)

        def polarization(pos):
            result = moments(pos)
            q = result['latent_charges']
            if self.les_arguments.get('remove_mean', True):
                counts = scatter_sum(mask, batch, dim=0, dim_size=n_graphs)
                means = scatter_sum(q, batch, dim=0, dim_size=n_graphs) / jnp.maximum(
                    counts, 1
                )
                q = (q - means[batch]) * mask
            frac = jnp.einsum('ni,nij->nj', pos, jnp.linalg.inv(cell)[batch])
            phase = jnp.exp(2j * jnp.pi * frac)
            return jnp.sum(
                q[:, None]
                * jnp.where(periodic[batch, None], phase / (2j * jnp.pi), pos),
                axis=0,
            )

        pos = data['positions']
        jac = jax.jacrev(lambda p: polarization(p).real)(pos) + 1j * jax.jacrev(
            lambda p: polarization(p).imag
        )(pos)
        frac = jnp.einsum('ni,nij->nj', pos, jnp.linalg.inv(cell)[batch])
        phase = jnp.where(periodic[batch, None], jnp.exp(2j * jnp.pi * frac), 1)
        bec = (jac.transpose(1, 0, 2) * phase.conj()[:, :, None]).real
        bec = jnp.einsum('nij,njk->nik', cell[batch].swapaxes(-1, -2), bec)
        if self.bec_output_index is not None:
            bec = bec[:, self.bec_output_index]
        if self.use_dipoles or self.use_induced_dipoles:
            jac_u = jax.jacrev(lambda p: moments(p)['latent_dipoles'].sum(axis=0))(
                pos
            ).transpose(1, 0, 2)
            if self.bec_output_index is not None:
                jac_u = jac_u[:, self.bec_output_index]
            bec = jnp.stack((bec, jac_u), axis=1)
        return bec * np.sqrt(self.les_arguments.get('epsilon_factor', 1.0))
