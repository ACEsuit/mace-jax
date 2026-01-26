from typing import Any

import jax.nn as jnn
import jax.numpy as jnp
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_register_module


class _ContinuousEmbed(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool, *, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(in_dim, out_dim, use_bias=use_bias, rngs=rngs)
        self.lin2 = nnx.Linear(out_dim, out_dim, use_bias=use_bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.lin1(x)
        x = jnn.silu(x)
        return self.lin2(x)


@nxx_register_module('mace.modules.embeddings.GenericJointEmbedding')
class GenericJointEmbedding(nnx.Module):
    """Flax version of the generic joint embedding fusion block."""

    base_dim: int
    embedding_specs: dict[str, Any] | None
    out_dim: int | None = None

    def __init__(
        self,
        base_dim: int,
        embedding_specs: dict[str, Any] | None,
        out_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.base_dim = base_dim
        self.embedding_specs = embedding_specs
        self.out_dim = out_dim
        if self.embedding_specs is None:
            raise ValueError('embedding_specs must be provided for joint embedding.')

        self.specs = {name: dict(spec) for name, spec in self.embedding_specs.items()}
        if not self.specs:
            raise ValueError('embedding_specs must contain at least one feature.')

        self.feature_names = tuple(self.specs.keys())
        self._out_dim = self.out_dim or self.base_dim
        self.total_dim = sum(spec['emb_dim'] for spec in self.specs.values())

        self.embedders = nnx.Dict()
        for name in self.feature_names:
            spec = self.specs[name]
            if spec['type'] == 'categorical':
                self.embedders[name] = nnx.Embed(
                    num_embeddings=spec['num_classes'],
                    features=spec['emb_dim'],
                    rngs=rngs,
                )
            elif spec['type'] == 'continuous':
                use_bias = spec.get('use_bias', True)
                in_dim = int(spec.get('in_dim', 1))
                self.embedders[name] = _ContinuousEmbed(
                    in_dim=in_dim,
                    out_dim=spec['emb_dim'],
                    use_bias=use_bias,
                    rngs=rngs,
                )
            else:
                raise ValueError(f'Unknown feature type {spec["type"]!r}')

        self.proj_lin = nnx.Linear(
            self.total_dim, self._out_dim, use_bias=False, rngs=rngs
        )

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
                emb = self.embedders[name](feat)
            elif spec['type'] == 'continuous':
                emb = self.embedders[name](feat)
            else:
                raise ValueError(f'Unknown feature type {spec["type"]!r}')

            embs.append(emb)

        if not embs:
            raise ValueError('No embeddings constructed; check embedding_specs input.')

        x = jnp.concatenate(embs, axis=-1)  # [N_nodes, total_dim]
        x = self.proj_lin(x)
        return jnn.silu(x)

    @classmethod
    def import_from_torch(cls, torch_module, variables):
        params = variables

        def assign(scope: str, key: str, value):
            node = params
            for part in scope.split('/'):
                if not part:
                    continue
                if part.isdigit():
                    part = int(part)
                if part not in node:
                    raise KeyError(f'Unknown NNX parameter scope {scope!r}')
                node = node[part]
            if key not in node:
                raise KeyError(f'Unknown NNX parameter key {key!r} at {scope!r}')
            node[key] = jnp.asarray(value, dtype=node[key].dtype)

        for name, spec in torch_module.specs.items():
            if spec['type'] == 'categorical':
                embed = torch_module.embedders[name]
                assign(
                    f'embedders/{name}',
                    'embedding',
                    embed.weight.detach().cpu().numpy(),
                )
            elif spec['type'] == 'continuous':
                seq = torch_module.embedders[name]
                lin1 = seq[0]
                lin2 = seq[2]
                assign(
                    f'embedders/{name}/lin1',
                    'kernel',
                    lin1.weight.detach().cpu().numpy().T,
                )
                if lin1.bias is not None:
                    assign(
                        f'embedders/{name}/lin1',
                        'bias',
                        lin1.bias.detach().cpu().numpy(),
                    )
                assign(
                    f'embedders/{name}/lin2',
                    'kernel',
                    lin2.weight.detach().cpu().numpy().T,
                )
                if lin2.bias is not None:
                    assign(
                        f'embedders/{name}/lin2',
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

        return params
