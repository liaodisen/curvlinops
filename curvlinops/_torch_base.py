"""Defines a minimal ``LinearOperator`` interface in PyTorch."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy
from scipy.sparse.linalg import LinearOperator
from torch import Size, Tensor, cat, device, dtype, from_numpy


class PyTorchLinearOperator:
    """Interface for linear operators in PyTorch.

    Heavily inspired by the Scipy interface
    (https://github.com/scipy/scipy/blob/v1.13.1/scipy/sparse/linalg/_interface.py),
    but only supports a sub-set of the functionality.

    The main difference is that the linear operator is defined over lists of tensors,
    rather than a single tensor. This is because in PyTorch, the space a linear operator
    acts on is a tensor product space, and not a single vector/matrix space. As a con-
    sequence, this linear operator interface allows multiplication onto vectors/matrices
    specified either in list format (as is standard when working with neural networks)
    or as a single flattened and concatenated PyTorch tensor (as is standard for SciPy
    linear operators).
    """

    def __init__(
        self, in_shape: List[Tuple[int, ...]], out_shape: List[Tuple[int, ...]]
    ):
        """Store the linear operator's input and output space dimensions.

        Args:
            in_shape: A list of shapes specifying the linear operator's input space.
            out_shape: A list of shapes specifying the linear operator's output space.
        """
        self._in_shape = [Size(s) for s in in_shape]
        self._out_shape = [Size(s) for s in out_shape]

        self._in_shape_flat = [s.numel() for s in self._in_shape]
        self._out_shape_flat = [s.numel() for s in self._out_shape]
        self.shape = (sum(self._out_shape_flat), sum(self._in_shape_flat))

    def __matmul__(self, X: Union[List[Tensor], Tensor]) -> Union[List[Tensor], Tensor]:
        """Multiply onto a vector or matrix given as PyTorch tensor or tensor list.

        Args:
            X: A vector or matrix to multiply onto, represented as a single tensor or a
                tensor list.

                Assume the linear operator has total shape ``[M, N]``:
                If ``X`` is a single tensor, it can be of shape ``[N, K]`` (matrix), or
                ``[N]`` (vector). The result will have shape ``[M, K]`` or ``[M]``.

                Instead, we can also pass ``X`` as tensor list:
                Assume the linear operator's rows are formed by a list of shapes
                ``[M1, M2, ...]`` and the columns by ``[N1, N2, ...]``, such that
                ``M1.numel() + M2.numel() + ... = M`` and ``N1.numel() + N2.numel() +
                ... = N``. Then, ``X`` can also be a list of tensors with shape
                ``[*N1], [*N2], ...`` (vector) or ``[*N1, K], [*N2, K], ...`` (matrix).
                In this case, the output will be tensor list with shapes ``[*M1], [*M2],
                ...`` (vector) or ``[K, *M1], [K, *M2], ...`` (matrix).

        Returns:
            The result of the matrix-vector or matrix-matrix multiplication in the same
            format as ``X``.
        """
        # convert to tensor list format
        X, list_format, is_vec, num_vecs = self._check_input_and_preprocess(X)

        # matrix-matrix-multiply using tensor list format
        AX = self._matmat(X)

        # return same format as ``X`` passed by the user
        return self._check_output_and_postprocess(AX, list_format, is_vec, num_vecs)

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

    def _check_input_and_preprocess(
        self, X: Union[List[Tensor], Tensor]
    ) -> Tuple[List[Tensor], bool, bool, int]:
        """Check input format and pre-process it to a matrix in tensor list format.

        Args:
            X: The object onto which the linear operator is multiplied.

        Returns:
            X_tensor_list: The input object in tensor list format.
            list_format: Whether the input was specified in tensor list format.
                This is useful for post-processing the multiplication's result.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.
        """
        if isinstance(X, Tensor):
            list_format = False
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_and_preprocess(X)

        elif isinstance(X, list) and all(isinstance(x, Tensor) for x in X):
            list_format = True
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_list_and_preprocess(X)

        else:
            raise ValueError(f"Input must be tensor or list of tensors. Got {type(X)}.")

        return X_tensor_list, list_format, is_vec, num_vecs

    def __check_tensor_and_preprocess(
        self, X: Tensor
    ) -> Tuple[List[Tensor], bool, int]:
        """Check single-tensor input format and process into a matrix tensor list.

        Args:
            X: The tensor onto which the linear operator is multiplied.

        Returns:
            X_processed: The input tensor as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """
        if X.ndim > 2 and X.shape[0] != self.shape[1]:
            raise ValueError(
                f"Input tensor must have shape ({self.shape[1]},) or "
                + f"({self.shape[1]}, K), with K arbitrary. Got {X.shape}."
            )

        # determine whether the input is a vector or matrix
        is_vec = X.ndim == 1
        num_vecs = 1 if is_vec else X.shape[1]

        # convert to matrix in tensor list format
        X_processed = [
            x.reshape(*s, num_vecs)
            for x, s in zip(X.split(self._in_shape_flat), self._in_shape)
        ]

        return X_processed, is_vec, num_vecs

    def __check_tensor_list_and_preprocess(
        self, X: List[Tensor]
    ) -> Tuple[List[Tensor], bool, int]:
        """Check tensor list input format and process into a matrix tensor list.

        Args:
            X: The tensor list onto which the linear operator is multiplied.

        Returns:
            X_processed: The input as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the tensor entries in the list have invalid shapes.
        """
        if len(X) != len(self._in_shape):
            raise ValueError(
                f"List must contain {len(self._in_shape)} tensors. Got {len(X)}."
            )

        # check if input is a vector or a matrix
        if all(x.shape == s for x, s in zip(X, self._in_shape)):
            is_vec, num_vecs = True, 1
        elif (
            all(
                x.ndim == len(s) + 1 and x.shape[:-1] == s
                for x, s in zip(X, self._in_shape)
            )
            and len({x.shape[-1] for x in X}) == 1
        ):
            is_vec, (num_vecs,) = False, {x.shape[-1] for x in X}
        else:
            raise ValueError(
                f"Input list must contain tensors with shapes {self._in_shape} "
                + "and optional trailing dimension for the matrix columns. "
                + f"Got {[x.shape for x in X]}."
            )

        # convert to matrix in tensor list format
        X_processed = [x.unsqueeze(-1) for x in X] if is_vec else X

        return X_processed, is_vec, num_vecs

    def _check_output_and_postprocess(
        self, AX: List[Tensor], list_format: bool, is_vec: bool, num_vecs: int
    ) -> Union[List[Tensor], Tensor]:
        """Check multiplication output and post-process it to the original format.

        Args:
            AX: The output of the multiplication as matrix in tensor list format.
            list_format: Whether the output should be in tensor list format.
            is_vec: Whether the output should be a vector or a matrix.
            num_vecs: The number of vectors represented by the output.

        Returns:
            AX_processed: The output in the original format, either as single tensor
                or list of tensors.

        Raises:
            ValueError: If the output tensor list has an invalid length or shape.
        """
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
            AX_processed = [Ax.squeeze(-1) for Ax in AX] if is_vec else AX
        else:
            AX_processed = cat(
                [Ax.reshape(s, num_vecs) for Ax, s in zip(AX, self._out_shape_flat)]
            )
            AX_processed = AX_processed.squeeze(-1) if is_vec else AX_processed

        return AX_processed

    ###############################################################################
    #                                 SCIPY EXPORT                                #
    ###############################################################################

    def to_scipy(self, dtype: Optional[numpy.dtype] = None) -> LinearOperator:
        """Wrap the PyTorch linear operator with a SciPy linear operator.

        Args:
            dtype: The data type of the SciPy linear operator. If ``None``, uses
                NumPy's default data dtype.


        Returns:
            A SciPy linear operator that carries out the matrix-vector products
            in PyTorch.
        """
        dev = self._infer_device()
        dt = self._infer_dtype()

        scipy_matmat = self._scipy_compatible(self.__matmul__, dev, dt)
        A_adjoint = self.adjoint()
        scipy_rmatmat = A_adjoint._scipy_compatible(A_adjoint.__matmul__, dev, dt)

        return LinearOperator(
            self.shape,
            matvec=scipy_matmat,
            rmatvec=scipy_rmatmat,
            matmat=scipy_matmat,
            rmatmat=scipy_rmatmat,
            dtype=numpy.dtype(dtype) if dtype is None else dtype,
        )

    def _infer_device(self) -> device:
        """Infer the linear operator's device.

        Returns:
            The device of the linear operator.

        Raises: # noqa: D402
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _infer_dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns:
            The data type of the linear operator.

        Raises: # noqa: D402
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _scipy_compatible(
        f: Callable[[Tensor], Tensor], device: device, dtype: dtype
    ) -> Callable[[numpy.ndarray], numpy.ndarray]:
        """Wrap a PyTorch matrix multiplication function to be compatible with SciPy.

        Args:
            f: The PyTorch matrix multiplication function.
            device: The device on which the PyTorch linear operator is defined.
            dtype: The data type of the PyTorch linear operator.

        Returns:
            A function that takes a NumPy array and returns a NumPy array.
        """

        def f_scipy(X: numpy.ndarray) -> numpy.ndarray:
            """Scipy-compatible matrix multiplication function.

            Args:
                X: The input matrix in NumPy format.

            Returns:
                The output matrix in NumPy format.
            """
            X_dtype = X.dtype
            X_torch = from_numpy(X).to(device, dtype)
            AX_torch = f(X_torch)
            return AX_torch.detach().cpu().numpy().astype(X_dtype)

        return f_scipy


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
