"""Contains LinearOperator implementation of the Hessian."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Tuple, Union

from backpack.hessianfree.hvp import hessian_vector_product
from torch import Tensor, zeros_like
from torch.autograd import grad
from torch.nn import Parameter

from curvlinops._base import CurvatureLinearOperator
from curvlinops.utils import split_list


class HessianLinearOperator(CurvatureLinearOperator):
    r"""Linear operator for the Hessian of an empirical risk in PyTorch.

    Consider the empirical risk

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        c \sum_{n=1}^{N}
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)

    with :math:`c = \frac{1}{N}` for ``reduction='mean'`` and :math:`c=1` for
    ``reduction='sum'``. The Hessian matrix is

    .. math::
        \nabla^2_{\mathbf{\theta}} \mathcal{L}
        =
        c \sum_{n=1}^{N}
        \nabla_{\mathbf{\theta}}^2
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)\,.

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports block operations.
            Default is ``True``.
    """

    SUPPORTS_BLOCKS: bool = True

    def __init__(
        self,
        model_func: Callable[[Tensor | MutableMapping], Tensor],
        loss_func: Callable[[Tensor, Tensor], Tensor] | None,
        params: List[Tensor | Parameter],
        data: Iterable[Tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        num_data: int | None = None,
        in_blocks: List[int] | None = None,
        out_blocks: List[int] | None = None,
        batch_size_fn: Callable[[Tensor | MutableMapping], int] | None = None,
    ):
        in_shape = out_shape = [tuple(p.shape) for p in params]
        (dt,) = {p.dtype for p in params}
        (dev,) = {p.device for p in params}
        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            in_shape,
            out_shape,
            dt,
            dev,
            progressbar=progressbar,
            num_data=num_data,
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            batch_size_fn=batch_size_fn,
        )

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch Hessian to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in list format.
                Tensors have same shape as trainable model parameters, and an
                additional leading axis for the matrix columns.

        Returns:
            Result of Hessian multiplication in list format. Has the same shape as
            ``M_list``, i.e. each tensor in the list has the shape of a parameter and a
            leading dimension of matrix columns.
        """
        assert self._loss_func is not None
        loss = self._loss_func(self._model_func(X), y)

        # Re-cycle first backward pass from the HVP's double-backward
        grad_params = list(grad(loss, self._params, create_graph=True))

        (num_vecs,) = {m.shape[-1] for m in M}
        result = [zeros_like(m) for m in M]

        assert self._in_blocks == self._out_blocks

        # per-block HMP
        for M_block, p_block, g_block, res_block in zip(
            split_list(M, self._in_blocks),
            split_list(self._params, self._in_blocks),
            split_list(grad_params, self._in_blocks),
            split_list(result, self._in_blocks),
        ):
            for n in range(num_vecs):
                col_n = hessian_vector_product(
                    loss, p_block, [m[..., n] for m in M_block], grad_params=g_block
                )
                for p, col in enumerate(col_n):
                    res_block[p][..., n].add_(col)

        return result

    def _adjoint(self) -> HessianLinearOperator:
        """Return the linear operator representing the adjoint.

        The Hessian is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
