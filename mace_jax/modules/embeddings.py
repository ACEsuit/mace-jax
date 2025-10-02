from typing import Any, Optional

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

from mace_jax.haiku.torch import (
    auto_import_from_torch,
    register_import,
)


@register_import('mace.modules.blocks.GenericJointEmbedding')
@auto_import_from_torch(separator='~')
class GenericJointEmbedding(hk.Module):
    """
    JAX/Haiku version of GenericJointEmbedding.
    Concat-fusion of node-/graph-level features with a base embedding.
    """

    def __init__(
        self,
        *,
        base_dim: int,
        embedding_specs: Optional[dict[str, Any]],
        out_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.base_dim = base_dim
        self.specs = dict(embedding_specs.items())
        self.out_dim = out_dim or base_dim

        # Build embedders (registered as submodules for parameter import)
        self.embedders = {}
        for feat_name, spec in self.specs.items():
            E = spec['emb_dim']
            use_bias = spec.get('use_bias', True)

            if spec['type'] == 'categorical':
                self.embedders[feat_name] = hk.Embed(
                    vocab_size=spec['num_classes'],
                    embed_dim=E,
                    name=f'{feat_name}_embed',
                )
            elif spec['type'] == 'continuous':
                self.embedders[feat_name] = hk.Sequential(
                    [
                        hk.Linear(E, with_bias=use_bias, name=f'{feat_name}_lin1'),
                        jnn.silu,
                        hk.Linear(E, with_bias=use_bias, name=f'{feat_name}_lin2'),
                    ],
                    name=f'{feat_name}_mlp',
                )
            else:
                raise ValueError(f'Unknown type {spec["type"]} for feature {feat_name}')

        # Concat → Linear(total_dim→out_dim) → SiLU
        self.total_dim = sum(spec['emb_dim'] for spec in self.specs.values())
        self.project = hk.Sequential(
            [
                hk.Linear(self.out_dim, with_bias=False, name='proj_lin'),
                jnn.silu,
            ],
            name='project',
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
        embs = []
        for name, spec in self.specs.items():
            feat = features[name]
            if spec['per'] == 'graph':
                feat = feat[batch][..., None]  # upsample to node level

            if spec['type'] == 'categorical':
                offset = spec.get('offset', 0)
                feat = (feat + offset).astype(jnp.int32).squeeze(-1)
                emb = self.embedders[name](feat)
            elif spec['type'] == 'continuous':
                emb = self.embedders[name](feat)
            else:
                raise ValueError(f'Unknown feature type {spec["type"]}')

            embs.append(emb)

        x = jnp.concatenate(embs, axis=-1)  # [N_nodes, total_dim]
        return self.project(x)  # [N_nodes, out_dim]
