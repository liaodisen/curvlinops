"""Contains functionality to analyze Hessian & GGN via matrix-free multiplication."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Optional, Tuple, Union
from warnings import warn

from einops import rearrange
from numpy import allclose, argwhere, float32, isclose, logical_not, ndarray
from numpy.random import rand
from scipy.sparse.linalg import LinearOperator
from torch import Size, Tensor
from torch import allclose as torch_allclose
from torch import argwhere as torch_argwhere
from torch import cat
from torch import device
from torch import device as torch_device
from torch import dtype, from_numpy
from torch import isclose as torch_isclose
from torch import logical_not as torch_logical_not
from torch import rand as torch_rand
from torch import tensor, zeros_like
from torch.autograd import grad
from torch.nn import Module, Parameter
from tqdm import tqdm


class _LinearOperator(LinearOperator):
    """Base class for linear operators of DNN matrices.

    Can be used with SciPy.

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports multiplication with
            a block-diagonal approximation rather than the full matrix.
            Default: ``False``.
    """

    SUPPORTS_BLOCKS: bool = False

    def __init__(
        self,
        model_func: Callable[[Tensor], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        shape: Optional[Tuple[int, int]] = None,
        num_data: Optional[int] = None,
        block_sizes: Optional[List[int]] = None,
        batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
    ):
        """Linear operator for DNN matrices.

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: A function that maps the mini-batch input X to predictions.
                Could be a PyTorch module representing a neural network.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value. If ``None``, there is no loss function and the
                represented matrix is independent of the loss function.
            params: List of differentiable parameters used by the prediction function.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function.
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.
            shape: Shape of the represented matrix. If ``None`` assumes ``(D, D)``
                where ``D`` is the total number of parameters
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            block_sizes: This argument will be ignored if the linear operator does not
                support blocks. List of integers indicating the number of
                ``nn.Parameter``s forming a block. Entries must sum to ``len(params)``.
                For instance ``[len(params)]`` considers the full matrix, while
                ``[1, 1, ...]`` corresponds to a block diagonal approximation where
                each parameter forms its own block.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.

        Raises:
            RuntimeError: If the check for deterministic behavior fails.
            ValueError: If ``block_sizes`` is specified but the linear operator does not
                support blocks.
            ValueError: If the sum of blocks does not equal the number of parameters.
            ValueError: If any block size is not positive.
            ValueError: If ``X`` is not a tensor and ``batch_size_fn`` is not specified.
        """
        if isinstance(next(iter(data))[0], MutableMapping) and batch_size_fn is None:
            raise ValueError(
                "When using dict-like custom data, `batch_size_fn` is required."
            )

        if shape is None:
            dim = sum(p.numel() for p in params)
            shape = (dim, dim)
        super().__init__(shape=shape, dtype=float32)

        self._params = params
        if block_sizes is not None:
            if not self.SUPPORTS_BLOCKS:
                raise ValueError(
                    "Block sizes were specified but operator does not support blocking."
                )
            if sum(block_sizes) != len(params):
                raise ValueError("Sum of blocks must equal the number of parameters.")
            if any(s <= 0 for s in block_sizes):
                raise ValueError("Block sizes must be positive.")
        self._block_sizes = [len(params)] if block_sizes is None else block_sizes

        self._model_func = model_func
        self._loss_func = loss_func
        self._data = data
        self.device = self._infer_device(self._params)
        self._progressbar = progressbar
        self._batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )

        self._N_data = (
            sum(
                self._batch_size_fn(X)
                for (X, _) in self._loop_over_data(desc="_N_data")
            )
            if num_data is None
            else num_data
        )

        if check_deterministic:
            old_device = self.device
            self.to_device(torch_device("cpu"))
            try:
                self._check_deterministic()
            except RuntimeError as e:
                raise e
            finally:
                self.to_device(old_device)

    @staticmethod
    def _infer_device(params: List[Parameter]) -> torch_device:
        """Infer the device on which to carry out matvecs.

        Args:
            params: DNN parameters that define the linear operators.

        Returns:
            Inferred device.

        Raises:
            RuntimeError: If the device cannot be inferred.
        """
        devices = {p.device for p in params}
        if len(devices) != 1:
            raise RuntimeError(f"Could not infer device. Parameters live on {devices}.")
        return devices.pop()

    def to_device(self, device: torch_device):
        """Load linear operator to a device (inplace).

        Args:
            device: Target device.
        """
        self.device = device

        if isinstance(self._model_func, Module):
            self._model_func = self._model_func.to(self.device)
        self._params = [p.to(device) for p in self._params]

        if isinstance(self._loss_func, Module):
            self._loss_func = self._loss_func.to(self.device)

    def _check_deterministic(self):
        """Check that the Linear operator is deterministic.

        Non-deterministic behavior is detected if:

        - Two independent applications of matvec onto the same vector yield different
          results
        - Two independent loss/gradient computations yield different results

        Note:
            Deterministic checks should be performed on CPU. We noticed that even when
            it passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        Raises:
            RuntimeError: If non-deterministic behavior is detected.
        """
        v = rand(self.shape[1]).astype(self.dtype)
        mat_v1 = self @ v
        mat_v2 = self @ v

        rtol, atol = 5e-5, 1e-6
        if not allclose(mat_v1, mat_v2, rtol=rtol, atol=atol):
            self.print_nonclose(mat_v1, mat_v2, rtol, atol)
            raise RuntimeError("Check for deterministic matvec failed.")

        if self._loss_func is None:
            return

        # only carried out if there is a loss function
        grad1, loss1 = self.gradient_and_loss()
        grad1, loss1 = (
            self.flatten_and_concatenate(grad1).cpu().numpy(),
            loss1.cpu().numpy(),
        )

        grad2, loss2 = self.gradient_and_loss()
        grad2, loss2 = (
            self.flatten_and_concatenate(grad2).cpu().numpy(),
            loss2.cpu().numpy(),
        )

        if not allclose(loss1, loss2, rtol=rtol, atol=atol):
            self.print_nonclose(loss1, loss2, rtol, atol)
            raise RuntimeError("Check for deterministic loss failed.")

        if not allclose(grad1, grad2, rtol=rtol, atol=atol):
            self.print_nonclose(grad1, grad2, rtol, atol)
            raise RuntimeError("Check for deterministic gradient failed.")

    @staticmethod
    def print_nonclose(array1: ndarray, array2: ndarray, rtol: float, atol: float):
        """Check if the two arrays are element-wise equal within a tolerance and print
        the entries that differ.

        Args:
            array1: First array for comparison.
            array2: Second array for comparison.
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        if not allclose(array1, array2, rtol=rtol, atol=atol):
            nonclose_idx = logical_not(isclose(array1, array2, rtol=rtol, atol=atol))
            for idx, a1, a2 in zip(
                argwhere(nonclose_idx),
                array1[nonclose_idx].flatten(),
                array2[nonclose_idx].flatten(),
            ):
                print(f"at index {idx}: {a1:.5e} ≠ {a2:.5e}, ratio: {a1 / a2:.5e}")

    def _matmat(self, M: ndarray) -> ndarray:
        """Matrix-matrix multiplication.

        Args:
            M: Matrix for multiplication.

        Returns:
            Matrix-multiplication result ``mat @ M``.
        """
        M_list = self._preprocess(M)
        out_list = [zeros_like(M) for M in M_list]

        for X, y in self._loop_over_data(desc="_matmat"):
            normalization_factor = self._get_normalization_factor(X, y)
            for out, current in zip(out_list, self._matmat_batch(X, y, M_list)):
                out.add_(current, alpha=normalization_factor)

        return self._postprocess(out_list)

    def _matmat_batch(
        self, X: Tensor, y: Tensor, M_list: List[Tensor]
    ) -> Tuple[Tensor]:
        """Apply the mini-batch matrix to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M_list: Matrix in list format (same shape as trainable model parameters with
                additional leading dimension of size number of columns).

        Returns: # noqa: D402
           Result of matrix-multiplication in list format.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError

    def _preprocess(self, M: ndarray) -> List[Tensor]:
        """Convert numpy matrix to torch parameter list format.

        Args:
            M: Matrix for multiplication. Has shape ``[D, K]`` where ``D`` is the
                number of parameters, and ``K`` is the number of columns.

        Returns:
            Matrix in list format. Each entry has the same shape as a parameter with
            an additional leading dimension of size ``K`` for the columns, i.e.
            ``[(K,) + p1.shape), (K,) + p2.shape, ...]``.
        """
        if M.dtype != self.dtype:
            warn(
                f"Input matrix is {M.dtype}, while linear operator is {self.dtype}. "
                + f"Converting to {self.dtype}."
            )
            M = M.astype(self.dtype)
        num_vectors = M.shape[1]

        result = from_numpy(M).to(self.device)
        # split parameter blocks
        dims = [p.numel() for p in self._params]
        result = result.split(dims)
        # column-index first + unflatten parameter dimension
        shapes = [(num_vectors,) + p.shape for p in self._params]
        result = [res.T.reshape(shape) for res, shape in zip(result, shapes)]

        return result

    def _postprocess(self, M_list: List[Tensor]) -> ndarray:
        """Convert torch list format of a matrix to flat numpy matrix format.

        Args:
            M_list: Matrix in list format. Each entry has a leading dimension of size
                ``K``. The remaining dimensions are flattened and concatenated.

        Returns:
            Flat matrix. Has shape ``[D, K]`` where ``D`` is sum of flattened and
            concatenated dimensions over all list entries.
        """
        result = [rearrange(M, "k ... -> (...) k") for M in M_list]
        return cat(result, dim=0).cpu().numpy()

    def _loop_over_data(
        self, desc: Optional[str] = None, add_device_to_desc: bool = True
    ) -> Iterable[Tuple[Tensor, Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Args:
            desc: Description for the progress bar. Will be ignored if progressbar is
                disabled.
            add_device_to_desc: Whether to add the device to the description.
                Default: ``True``.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = iter(self._data)

        if self._progressbar:
            desc = f"{self.__class__.__name__}{'' if desc is None else f'.{desc}'}"
            if add_device_to_desc:
                desc = f"{desc} (on {str(self.device)})"
            data_iter = tqdm(data_iter, desc=desc)

        for X, y in data_iter:
            # Assume everything is handled by the model
            # if `X` is a custom data format
            if isinstance(X, Tensor):
                X = X.to(self.device)
            y = y.to(self.device)
            yield (X, y)

    def gradient_and_loss(self) -> Tuple[List[Tensor], Tensor]:
        """Evaluate the gradient and loss on the data.

        (Not really part of the LinearOperator interface.)

        Returns:
            Gradient and loss on the data set.

        Raises:
            ValueError: If there is no loss function.
        """
        if self._loss_func is None:
            raise ValueError("No loss function specified.")

        total_loss = tensor([0.0], device=self.device)
        total_grad = [zeros_like(p) for p in self._params]

        for X, y in self._loop_over_data(desc="gradient_and_loss"):
            loss = self._loss_func(self._model_func(X), y)
            normalization_factor = self._get_normalization_factor(X, y)

            for grad_param, current in zip(total_grad, grad(loss, self._params)):
                grad_param.add_(current, alpha=normalization_factor)
            total_loss.add_(loss.detach(), alpha=normalization_factor)

        return total_grad, total_loss

    def _get_normalization_factor(self, X: Tensor, y: Tensor) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor
        """
        return {"sum": 1.0, "mean": self._batch_size_fn(X) / self._N_data}[
            self._loss_func.reduction
        ]

    @staticmethod
    def flatten_and_concatenate(tensors: List[Tensor]) -> Tensor:
        """Flatten then concatenate all tensors in a list.

        Args:
            tensors: List of tensors.

        Returns:
            Concatenated flattened tensors.
        """
        return cat([t.flatten() for t in tensors])


class PyTorchLinearOperator:
    """Interface for linear operators in PyTorch.

    Heavily inspired by the Scipy interface
    (https://github.com/scipy/scipy/blob/v1.13.1/scipy/sparse/linalg/_interface.py),
    but only supports a sub-set of the functionality.

    The main difference is that the linear operator is defined over lists of tensors,
    rather than a single tensor. This is because in PyTorch, the space a linear operator
    acts on is a tensor product space, and not a single vector/matrix space.
    """

    def __init__(
        self,
        in_shape: List[Tuple[int, ...]],
        out_shape: List[Tuple[int, ...]],
        dt: dtype,
        dev: device,
    ):
        """Store shapes of the output and input space, data type, and device.

        Args:
            in_shape: A list of shapes specifying the tensor list format of the linear
                operator's range.
            out_shape: A list of shapes specifying the tensor list format of the linear
                operator's domain.
            dt: The data type of the linear operator.
            dev: The device the linear operator is stored on.
        """
        self._in_shape = [Size(s) for s in in_shape]
        self._out_shape = [Size(s) for s in out_shape]

        self._in_shape_flat = [s.numel() for s in self._in_shape]
        self._out_shape_flat = [s.numel() for s in self._out_shape]
        self.shape = (sum(self._out_shape_flat), sum(self._in_shape_flat))

        self.dtype = dt
        self.device = dev

    def __matmul__(self, X: Union[List[Tensor], Tensor]) -> Union[List[Tensor], Tensor]:
        """Multiply onto a vector or matrix given in list or tensor format.

        Args:
            X: A list of tensors or a tensor to multiply onto.
                Assume the linear operator has total shape ``[M, N]``.
                ``X`` can be of shape ``[N, K]`` (matrix), or ``[N]`` (vector), and the
                results will be of shape ``[M, K]`` or ``[M]``, respectively.

                Instead, we can also pass ``X`` in tensor list format.
                If the rows are formed by a list of shapes ``[M1, M2, ...]`` and the
                columns by ``[N1, N2, ...]``, such that
                ``M1.numel() + M2.numel() + ... = M`` and ``N1.numel() + N2.numel() +
                ... = N``, then the input can also be a list of tensors with shape
                ``[*N1], [*N2], ...`` (vector) or ``[*N1, K], [*N2, K], ...`` (matrix).
                In this case, the output will be a list of tensors with shape
                ``[*M1], [*M2], ...`` (vector case) or ``[K, *M1], [K, *M2], ...``
                (matrix case).

        Returns:
            The result of the matrix-vector or matrix-matrix multiplication in either
            list or tensor format (same as the input's format, see above).
        """
        # unify input format for `matmat`
        X, list_format, is_vec, num_vecs = self._preprocess_to_input_tensor_list(X)

        # matrix-matrix-multiplication in tensor list format
        AX = self._matmat(X)

        # return same format as passed by the user
        return self._postprocess_to_output(AX, list_format, is_vec, num_vecs)

    def _preprocess_to_input_tensor_list(
        self, X: Union[List[Tensor], Tensor]
    ) -> Tuple[List[Tensor], bool, bool, int]:
        # pre-processing: detect whether the multiplication argument is in list format
        # and whether it represents a single vector or a matrix, convert everything to
        # list format and treat vectors as matrices with a single column.

        # vector or matrix supplied as single tensor
        if isinstance(X, Tensor):
            list_format = False

            # check if input is a vector or matrix
            if X.ndim == 1 and X.shape[0] == self.shape[1]:
                is_vec = True
                num_vecs = 1
            elif X.ndim == 2 and X.shape[0] == self.shape[1]:
                is_vec = False
                num_vecs = X.shape[1]
            else:
                raise ValueError(
                    f"Input tensor must have shape ({self.shape[1]},) or "
                    + f"({self.shape[1]}, K), with K arbitrary. Got {X.shape}."
                )

            # convert to matrix in tensor list format
            if is_vec:
                X = X.unsqueeze(-1)

            X = [
                x.reshape(*s, num_vecs)
                for x, s in zip(X.split(self._in_shape_flat), self._in_shape)
            ]

        # vector or matrix supplied in tensor list format
        elif isinstance(X, list) and all(isinstance(x, Tensor) for x in X):
            list_format = True

            if len(X) != len(self._in_shape):
                raise ValueError(
                    f"Input list must contain {len(self._in_shape)} tensors. Got {len(X)}."
                )

            # check if input is a vector or a matrix
            if all(x.shape == s for x, s in zip(X, self._in_shape)):
                is_vec = True
                num_vecs = 1
            elif (
                all(
                    x.ndim == len(s) + 1 and x.shape[:-1] == s
                    for x, s in zip(X, self._in_shape)
                )
                and len({x.shape[-1] for x in X}) == 1
            ):
                is_vec = False
                (num_vecs,) = {x.shape[-1] for x in X}
            else:
                raise ValueError(
                    f"Input list must contain tensors with shapes {self._in_shape} "
                    + "and optional trailing dimension for the matrix columns. "
                    + f"Got {[x.shape for x in X]}."
                )

            # convert to matrix in tensor list format
            if is_vec:
                X = [x.unsqueeze(-1) for x in X]

        else:
            raise ValueError(f"Input must be tensor or list of tensors. Got {type(X)}.")

        return X, list_format, is_vec, num_vecs

    def _postprocess_to_output(
        self, AX: List[Tensor], list_format: bool, is_vec: bool, num_vecs: int
    ) -> Union[List[Tensor], Tensor]:
        # verify output tensor list format
        if len(AX) != len(self._out_shape):
            raise ValueError(
                f"Output list must contain {len(self._out_shape)} tensors. Got {len(AX)}."
            )
        if any(Ax.shape != (*s, num_vecs) for Ax, s in zip(AX, self._out_shape)):
            raise ValueError(
                f"Output tensors must have shapes {self._out_shape} and additional "
                + f"trailing dimension of {num_vecs}. "
                + f"Got {[Ax.shape for Ax in AX]}."
            )

        if list_format:
            if is_vec:
                AX = [Ax.squeeze(-1) for Ax in AX]
        else:
            AX = cat(
                [Ax.reshape(s, num_vecs) for Ax, s in zip(AX, self._out_shape_flat)]
            )
            if is_vec:
                AX = AX.squeeze(-1)

        return AX

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            X: A list of tensors representing the matrix to multiply onto.
                The list must contain tensors of shape ``[*N1, K], [*N2, K], ...``,
                where ``N1, N2, ...`` are the shapes of the linear operator's columns.

        Returns: # noqa: D402
            A list of tensors with shape ``[*M1, K], [*M2, K], ...``, where ``M1, M2,
            ...`` are the shapes of the linear operator's rows.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def adjoint(self) -> PyTorchLinearOperator:
        """Return the adjoint of the linear operator.

        Returns:
            The adjoint of the linear operator.
        """
        return self._adjoint()

    def _adjoint(self) -> PyTorchLinearOperator:
        """Adjoint of the linear operator.

        Returns: # noqa: D402
            The adjoint of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def scipy_compatible(
        self, f: Callable[[Tensor], Tensor]
    ) -> Callable[[ndarray], ndarray]:
        def f_scipy(X: ndarray) -> ndarray:
            X_dtype = X.dtype
            X_torch = from_numpy(X).to(self.device, self.dtype)
            AX_torch = f(X_torch)
            return AX_torch.detach().cpu().numpy().astype(X_dtype)

        return f_scipy

    def to_scipy(self) -> LinearOperator:
        """Wrap the PyTorch linear operator with a SciPy linear operator.

        Returns:
            A SciPy linear operator that carries out the matrix-vector products
            in PyTorch.
        """
        scipy_matmat = self.scipy_compatible(self.__matmul__)
        dt = float32
        A_adjoint = self.adjoint()
        scipy_rmatmat = A_adjoint.scipy_compatible(A_adjoint.__matmul__)

        return LinearOperator(
            self.shape,
            matvec=scipy_matmat,
            rmatvec=scipy_rmatmat,
            matmat=scipy_matmat,
            rmatmat=scipy_rmatmat,
            dtype=dt,
        )


class CurvatureLinearOperator(PyTorchLinearOperator):
    """Base class for a PyTorch linear operators representing curvature matrices.

    Handles accumulation of matrix-matrix-products over all mini-batches and restricting
    the matrix to blocks along the diagonal.

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports multiplication with
            a block-diagonal approximation rather than the full matrix.
            Default: ``False``.
    """

    SUPPORTS_BLOCKS = False

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Union[Tensor, Parameter]],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        in_shape: List[Tuple[int, ...]],
        out_shape: List[Tuple[int, ...]],
        dt: dtype,
        dev: device,
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        in_blocks: Optional[List[int]] = None,
        out_blocks: Optional[List[int]] = None,
        batch_size_fn: Optional[Callable[[Union[Tensor, MutableMapping]], int]] = None,
    ):
        in_shape = [tuple(p.shape) for p in params] if in_shape is None else in_shape
        out_shape = [tuple(p.shape) for p in params] if out_shape is None else out_shape
        super().__init__(in_shape, out_shape, dt, dev)

        if isinstance(next(iter(data))[0], MutableMapping) and batch_size_fn is None:
            raise ValueError(
                "When using dict-like custom data, `batch_size_fn` is required."
            )

        assert (in_blocks is None and out_blocks is None) or (
            in_blocks is not None and out_blocks is not None
        )
        if in_blocks is not None and out_blocks is not None:
            self._in_blocks = in_blocks
            self._out_blocks = out_blocks
            if not self.SUPPORTS_BLOCKS:
                raise ValueError(
                    "Block sizes were specified but operator does not support blocking."
                )
            if sum(self._out_blocks) != len(self._out_shape):
                raise ValueError("Sum of out-blocks must equal the number of outputs.")
            if sum(self._in_blocks) != len(self._in_shape):
                raise ValueError("Sum of in-blocks must equal the number of inputs.")
            if any(s <= 0 for s in self._out_blocks + self._in_blocks):
                raise ValueError("Block sizes must be positive.")
        else:
            self._out_blocks = [len(self._out_shape)]
            self._in_blocks = [len(self._in_shape)]

        self._model_func = model_func
        self._loss_func = loss_func
        self._params = params
        self._data = data
        self._progressbar = progressbar
        self._batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )
        self._N_data = (
            sum(
                self._batch_size_fn(X)
                for (X, _) in self._loop_over_data(desc="_N_data")
            )
            if num_data is None
            else num_data
        )

        if check_deterministic:
            old_device = self.device
            self.to_device(torch_device("cpu"))
            try:
                self._check_deterministic()
            except RuntimeError as e:
                raise e
            finally:
                self.to_device(old_device)

    def _check_deterministic(self):
        """Check that the Linear operator is deterministic.

        Non-deterministic behavior is detected if:

        - Two independent applications of matvec onto the same vector yield different
          results
        - Two independent loss/gradient computations yield different results

        Note:
            Deterministic checks should be performed on CPU. We noticed that even when
            it passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        Raises:
            RuntimeError: If non-deterministic behavior is detected.
        """
        v = torch_rand(self.shape[1], dtype=self.dtype, device=self.device)
        mat_v1 = self @ v
        mat_v2 = self @ v

        rtol, atol = 5e-5, 1e-6
        if not allclose(mat_v1, mat_v2, rtol=rtol, atol=atol):
            self.print_nonclose(mat_v1, mat_v2, rtol, atol)
            raise RuntimeError("Check for deterministic matvec failed.")

        if self._loss_func is None:
            return

        # only carried out if there is a loss function
        grad1, loss1 = self.gradient_and_loss()
        grad2, loss2 = self.gradient_and_loss()

        if not loss1.allclose(loss2, rtol=rtol, atol=atol):
            self.print_nonclose(loss1, loss2, rtol, atol)
            raise RuntimeError("Check for deterministic loss failed.")

        for g1, g2 in zip(grad1, grad2):
            if not g1.allclose(g2, rtol=rtol, atol=atol):
                self.print_nonclose(g1, g2, rtol, atol)
                raise RuntimeError("Check for deterministic gradient failed.")

    def gradient_and_loss(self) -> Tuple[List[Tensor], Tensor]:
        """Evaluate the gradient and loss on the data.

        (Not really part of the LinearOperator interface.)

        Returns:
            Gradient and loss on the data set.

        Raises:
            ValueError: If there is no loss function.
        """
        if self._loss_func is None:
            raise ValueError("No loss function specified.")

        total_loss = tensor([0.0], device=self.device)
        total_grad = [zeros_like(p) for p in self._params]

        for X, y in self._loop_over_data(desc="gradient_and_loss"):
            loss = self._loss_func(self._model_func(X), y)
            normalization_factor = self._get_normalization_factor(X, y)

            for grad_param, current in zip(total_grad, grad(loss, self._params)):
                grad_param.add_(current, alpha=normalization_factor)
            total_loss.add_(loss.detach(), alpha=normalization_factor)

        return total_grad, total_loss

    @staticmethod
    def print_nonclose(tensor1: Tensor, tensor2: Tensor, rtol: float, atol: float):
        """Check if the two tensors are element-wise equal within a tolerance.

        Print the entries that differ.

        Args:
            tensor1: First tensor for comparison.
            tensor2: Second tensor for comparison.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
        """
        if not torch_allclose(tensor1, tensor2, rtol=rtol, atol=atol):
            nonclose_idx = torch_logical_not(
                torch_isclose(tensor1, tensor2, rtol=rtol, atol=atol)
            )
            for idx, a1, a2 in zip(
                torch_argwhere(nonclose_idx),
                tensor1[nonclose_idx].flatten(),
                tensor2[nonclose_idx].flatten(),
            ):
                print(f"at index {idx}: {a1:.5e} ≠ {a2:.5e}, ratio: {a1 / a2:.5e}")

    def to_device(self, device: torch_device):
        """Load linear operator to a device (inplace).

        Args:
            device: Target device.
        """
        self.device = device

        if isinstance(self._model_func, Module):
            self._model_func = self._model_func.to(self.device)
        self._params = [p.to(device) for p in self._params]

        if isinstance(self._loss_func, Module):
            self._loss_func = self._loss_func.to(self.device)

    def _loop_over_data(
        self, desc: Optional[str] = None, add_device_to_desc: bool = True
    ) -> Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Args:
            desc: Description for the progress bar. Will be ignored if progressbar is
                disabled.
            add_device_to_desc: Whether to add the device to the description.
                Default: ``True``.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = iter(self._data)

        if self._progressbar:
            desc = f"{self.__class__.__name__}{'' if desc is None else f'.{desc}'}"
            if add_device_to_desc:
                desc = f"{desc} (on {str(self.device)})"
            data_iter = tqdm(data_iter, desc=desc)

        for X, y in data_iter:
            # Assume everything is handled by the model
            # if `X` is a custom data format
            if isinstance(X, Tensor):
                X = X.to(self.device)
            y = y.to(self.device)
            yield (X, y)

    def _get_normalization_factor(
        self, X: Union[Tensor, MutableMapping], y: Tensor
    ) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor
        """
        return {"sum": 1.0, "mean": self._batch_size_fn(X) / self._N_data}[
            self._loss_func.reduction
        ]

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        AM = [zeros_like(m) for m in M]

        for X, y in self._loop_over_data(desc="_matmat"):
            normalization_factor = self._get_normalization_factor(X, y)
            for Am, batch_Am in zip(AM, self._matmat_batch(X, y, M)):
                Am.add_(batch_Am, alpha=normalization_factor)

        return AM

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> Tuple[Tensor]:
        """Apply the mini-batch matrix to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix in list format.

        Returns: # noqa: D402
           Result of matrix-multiplication in list format.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError
