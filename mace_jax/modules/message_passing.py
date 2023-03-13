from typing import Callable, Optional, Sequence, Union

import e3nn_jax as e3nn
import jax.numpy as jnp
import haiku as hk


class MessagePassingConvolution(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps,
        activation: Callable,
        torch_style: bool = False,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.activation = activation
        self.torch_style = torch_style

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2
        assert edge_attrs.ndim == 2

        messages = node_feats[senders]

        if not self.torch_style:
            messages = e3nn.concatenate(
                [
                    messages.filter(self.target_irreps),
                    e3nn.tensor_product(
                        messages,
                        edge_attrs.filter(drop="0e"),
                        filter_ir_out=self.target_irreps,
                    ),
                ]
            ).regroup()  # [n_edges, irreps]
        else:
            one = e3nn.IrrepsArray.ones("0e", edge_attrs.shape[:-1])
            messages = e3nn.tensor_product(
                messages, e3nn.concatenate([one, edge_attrs.filter(drop="0e")])
            ).filter(self.target_irreps)

        mix = MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
            torch_style=self.torch_style,
        )(
            edge_attrs.filter(keep="0e")
        )  # [n_edges, num_irreps]

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)


class MultiLayerPerceptron(hk.Module):
    """Just a simple MLP for scalars. No equivariance here.
    Args:
        list_neurons (list of int): number of neurons in each layer (excluding the input layer)
        act (optional callable): activation function
        gradient_normalization (str or float): normalization of the gradient
            - "element": normalization done in initialization variance of the weights, (the default in pytorch)
                gives the same importance to each neuron, a layer with more neurons will have a higher importance
                than a layer with less neurons
            - "path" (default): normalization done explicitly in the forward pass,
                gives the same importance to every layer independently of the number of neurons
    """

    def __init__(
        self,
        list_neurons: Sequence[int],
        act: Optional[Callable],
        *,
        gradient_normalization: Union[str, float] = None,
        output_activation: Union[Callable, bool] = True,
        name: Optional[str] = None,
        torch_style: bool = False,
    ):
        super().__init__(name=name)

        self.list_neurons = list_neurons
        self.act = act

        if output_activation is True:
            self.output_activation = self.act
        elif output_activation is False:
            self.output_activation = None
        else:
            assert callable(output_activation)
            self.output_activation = output_activation

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]
        self.gradient_normalization = gradient_normalization
        self.torch_style = torch_style

    def __call__(
        self, x: Union[jnp.ndarray, e3nn.IrrepsArray]
    ) -> Union[jnp.ndarray, e3nn.IrrepsArray]:
        """Evaluate the MLP
        Input and output are either `jax.numpy.ndarray` or `IrrepsArray`.
        If the input is a `IrrepsArray`, it must contain only scalars.
        Args:
            x (IrrepsArray): input of shape ``[..., input_size]``
        Returns:
            IrrepsArray: output of shape ``[..., list_neurons[-1]]``
        """
        if isinstance(x, e3nn.IrrepsArray):
            if not x.irreps.is_scalar():
                raise ValueError("MLP only works on scalar (0e) input.")
            x = x.array
            output_irrepsarray = True
        else:
            output_irrepsarray = False
        if not self.torch_style:
            act = None if self.act is None else e3nn.normalize_function(self.act)
        else:
            act = lambda x: self.act(x) * 1.6791767923989418
        last_act = (
            None
            if self.output_activation is None
            else e3nn.normalize_function(self.output_activation)
        )
        for i, h in enumerate(self.list_neurons):
            alpha = 1 / x.shape[-1]
            d = hk.Linear(
                h,
                with_bias=False,
                w_init=hk.initializers.RandomNormal(
                    stddev=jnp.sqrt(alpha) ** (1.0 - self.gradient_normalization)
                ),
                name=f"linear_{i}",
            )
            x = jnp.sqrt(alpha) ** self.gradient_normalization * d(x)
            if i < len(self.list_neurons) - 1:
                if act is not None:
                    x = act(x)
            else:
                if last_act is not None:
                    x = last_act(x)

        if output_irrepsarray:
            x = e3nn.IrrepsArray(e3nn.Irreps(f"{x.shape[-1]}x0e"), x)
        return x
