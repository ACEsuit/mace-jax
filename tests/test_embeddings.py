import jax
import jax.numpy as jnp
import numpy as np
import torch
from mace.modules.embeddings import GenericJointEmbedding as TorchGenericJointEmbedding

from mace_jax.adapters.flax.torch import init_from_torch
from mace_jax.modules.embeddings import (
    GenericJointEmbedding as FlaxGenericJointEmbedding,
)


def _tf32_precision_in_use() -> bool:
    """Return ``True`` when GPU matmuls run with TF32 precision."""

    backend = jax.default_backend()
    precision = jax.config.jax_default_matmul_precision
    if backend != 'gpu':
        return False

    if precision is None:
        return True  # default precision on GPU implies TF32

    precision_str = str(precision).lower()
    return precision_str in {'tensorfloat32', 'fastest'}


class TestGenericJointEmbedding:
    """Compare the Flax joint embedding against the Torch reference."""

    def test_forward_matches_torch(self):
        rng = np.random.default_rng(0)

        embedding_specs = {
            'cat_node': {
                'type': 'categorical',
                'per': 'node',
                'emb_dim': 4,
                'num_classes': 5,
            },
            'cat_graph': {
                'type': 'categorical',
                'per': 'graph',
                'emb_dim': 2,
                'num_classes': 3,
            },
            'cont_node': {
                'type': 'continuous',
                'per': 'node',
                'emb_dim': 3,
                'in_dim': 2,
                'use_bias': True,
            },
            'cont_graph': {
                'type': 'continuous',
                'per': 'graph',
                'emb_dim': 2,
                'in_dim': 1,
                'use_bias': False,
            },
        }

        base_dim = 5
        out_dim = 7

        torch_module = TorchGenericJointEmbedding(
            base_dim=base_dim,
            embedding_specs=embedding_specs,
            out_dim=out_dim,
        )
        torch_module.float().eval()

        num_nodes = 6
        num_graphs = 2
        batch_np = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        features_np = {
            'cat_node': rng.integers(
                0,
                embedding_specs['cat_node']['num_classes'],
                size=(num_nodes, 1),
            ),
            'cat_graph': rng.integers(
                0,
                embedding_specs['cat_graph']['num_classes'],
                size=(num_graphs,),
            ),
            'cont_node': rng.standard_normal(
                (num_nodes, embedding_specs['cont_node']['in_dim']),
                dtype=np.float32,
            ),
            'cont_graph': rng.standard_normal(num_graphs, dtype=np.float32),
        }

        batch_torch = torch.tensor(batch_np, dtype=torch.long)
        features_torch = {
            'cat_node': torch.tensor(features_np['cat_node'], dtype=torch.long),
            'cat_graph': torch.tensor(features_np['cat_graph'], dtype=torch.long),
            'cont_node': torch.tensor(features_np['cont_node'], dtype=torch.float32),
            'cont_graph': torch.tensor(features_np['cont_graph'], dtype=torch.float32),
        }

        with torch.no_grad():
            out_torch = torch_module(batch_torch, features_torch).cpu().numpy()

        features_jax = {
            'cat_node': jnp.asarray(features_np['cat_node'], dtype=jnp.int32),
            'cat_graph': jnp.asarray(features_np['cat_graph'], dtype=jnp.int32),
            'cont_node': jnp.asarray(features_np['cont_node'], dtype=jnp.float32),
            'cont_graph': jnp.asarray(features_np['cont_graph'], dtype=jnp.float32),
        }

        flax_module = FlaxGenericJointEmbedding(
            base_dim=base_dim,
            embedding_specs=embedding_specs,
            out_dim=out_dim,
        )
        flax_module, variables = init_from_torch(
            flax_module,
            torch_module,
            jax.random.PRNGKey(0),
            jnp.asarray(batch_np),
            features_jax,
        )

        out_flax = flax_module.apply(variables, jnp.asarray(batch_np), features_jax)
        out_flax_np = np.asarray(out_flax)

        rtol = 1e-5
        atol = 1e-6
        if _tf32_precision_in_use():
            rtol = 2e-2
            atol = 5e-4

        np.testing.assert_allclose(out_flax_np, out_torch, rtol=rtol, atol=atol)
