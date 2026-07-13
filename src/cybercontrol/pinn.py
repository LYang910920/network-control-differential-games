"""Physics-informed neural-network primitives shared by Note 2 methods."""

from __future__ import annotations


def positive(raw, eps: float = 1e-6):
    """Map unconstrained tensors to positive rates."""

    import torch

    return torch.nn.functional.softplus(raw) + eps


def time_derivative(values, time):
    """Differentiate every flattened output channel with respect to time.

    ``values`` may be ``[batch, channels]`` or ``[batch, nodes, channels]``.
    The returned tensor has the same shape.
    """

    import torch

    original_shape = values.shape
    flat = values.reshape(values.shape[0], -1)
    derivative = torch.cat(
        [
            torch.autograd.grad(flat[:, j].sum(), time, create_graph=True)[0]
            for j in range(flat.shape[1])
        ],
        dim=1,
    )
    return derivative.reshape(original_shape)


def residual_mse(residual):
    """Mean squared residual with a finite-value guard."""

    import torch

    if not torch.isfinite(residual).all():
        raise FloatingPointError("PINN residual contains non-finite values")
    return residual.square().mean()


def masked_mse(prediction, target, mask=None):
    """MSE over all entries or a validated boolean observation mask."""

    if mask is None:
        return (prediction - target).square().mean()
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if selected_prediction.numel() == 0:
        raise ValueError("observation mask selects no values")
    return (selected_prediction - selected_target).square().mean()
