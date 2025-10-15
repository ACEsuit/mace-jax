from typing import Any, Optional

import jax.nn as jnn
import jax.numpy as jnp
from flax import linen as fnn
from flax.core import freeze, unfreeze

from mace_jax.adapters.flax.torch import register_flax_module


@register_flax_module('mace.modules.embeddings.GenericJointEmbedding')
class GenericJointEmbedding(fnn.Module):
    """Flax version of the generic joint embedding fusion block."""

    base_dim: int
    embedding_specs: dict[str, Any] | None
    out_dim: int | None = None

    def setup(self) -> None:
        if self.embedding_specs is None:
            raise ValueError('embedding_specs must be provided for joint embedding.')

        self.specs = {name: dict(spec) for name, spec in self.embedding_specs.items()}
        if not self.specs:
            raise ValueError('embedding_specs must contain at least one feature.')

        self.feature_names = tuple(self.specs.keys())
        self._out_dim = self.out_dim or self.base_dim
        self.total_dim = sum(spec['emb_dim'] for spec in self.specs.values())

    @fnn.compact
    def __call__(
        self,
        batch: jnp.ndarray,  # [N_nodes,] graph indices
        features: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        features[name] is either [N_graphs, …] or [N_nodes, …].
        Per-graph features are upsampled via batch indexing.
        Returns: [N_nodes, out_dim]
        """

        embs: list[jnp.ndarray] = []
        for name in self.feature_names:
            if name not in features:
                raise KeyError(f'Missing feature {name!r} required by joint embedding.')

            spec = self.specs[name]
            feat = features[name]

            per = spec.get('per', 'node')
            if per == 'graph':
                feat = feat[batch][..., None]
            elif per != 'node':
                raise ValueError(f"Unknown 'per' value {per!r} for feature {name!r}.")

            if spec['type'] == 'categorical':
                offset = spec.get('offset', 0)
                feat = (feat + offset).astype(jnp.int32).squeeze(-1)
                emb = fnn.Embed(
                    num_embeddings=spec['num_classes'],
                    features=spec['emb_dim'],
                    name=f'{name}_embed',
                )(feat)
            elif spec['type'] == 'continuous':
                use_bias = spec.get('use_bias', True)
                emb = fnn.Dense(
                    spec['emb_dim'],
                    use_bias=use_bias,
                    name=f'{name}_lin1',
                )(feat)
                emb = jnn.silu(emb)
                emb = fnn.Dense(
                    spec['emb_dim'],
                    use_bias=use_bias,
                    name=f'{name}_lin2',
                )(emb)
            else:
                raise ValueError(f'Unknown feature type {spec["type"]!r}')

            embs.append(emb)

        if not embs:
            raise ValueError('No embeddings constructed; check embedding_specs input.')

        x = jnp.concatenate(embs, axis=-1)  # [N_nodes, total_dim]
        x = fnn.Dense(self._out_dim, use_bias=False, name='proj_lin')(x)
        return jnn.silu(x)

    @classmethod
    def import_from_torch(cls, torch_module, flax_variables):
        variables = unfreeze(flax_variables)
        params = variables.setdefault('params', {})

        def assign(scope: str, key: str, value):
            target = params.get(scope)
            if target is None:
                raise KeyError(f'Unknown Flax parameter scope {scope!r}')
            target[key] = jnp.asarray(value, dtype=target[key].dtype)

        for name, spec in torch_module.specs.items():
            if spec['type'] == 'categorical':
                embed = torch_module.embedders[name]
                assign(
                    f'{name}_embed',
                    'embedding',
                    embed.weight.detach().cpu().numpy(),
                )
            elif spec['type'] == 'continuous':
                seq = torch_module.embedders[name]
                lin1 = seq[0]
                lin2 = seq[2]
                assign(
                    f'{name}_lin1',
                    'kernel',
                    lin1.weight.detach().cpu().numpy().T,
                )
                if lin1.bias is not None:
                    assign(
                        f'{name}_lin1',
                        'bias',
                        lin1.bias.detach().cpu().numpy(),
                    )
                assign(
                    f'{name}_lin2',
                    'kernel',
                    lin2.weight.detach().cpu().numpy().T,
                )
                if lin2.bias is not None:
                    assign(
                        f'{name}_lin2',
                        'bias',
                        lin2.bias.detach().cpu().numpy(),
                    )
            else:
                raise ValueError(f'Unknown feature type {spec["type"]!r}')

        proj = torch_module.project[0]
        assign(
            'proj_lin',
            'kernel',
            proj.weight.detach().cpu().numpy().T,
        )

        return freeze(variables)
