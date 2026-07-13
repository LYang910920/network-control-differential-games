"""Configurable neural blocks and architecture summaries.

The module is optional: importing it requires PyTorch, while importing the
base :mod:`cybercontrol` package does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Mapping

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Install the optional Torch dependency with `pip install -e .[torch]`."
    ) from exc


@dataclass(frozen=True)
class ArchitectureConfig:
    """Inspectable architecture choices shared by RL and PINN runners."""

    name: str
    input_dim: int
    output_dim: int
    width: int = 64
    depth: int = 2
    activation: str = "tanh"
    normalization: str = "none"
    encoder: str = "mlp"
    decoder: str = "linear"
    fourier_features: int = 0
    recurrent: bool = False


@dataclass(frozen=True)
class ArchitectureDescriptor:
    """Human- and manifest-readable description of a registered architecture."""

    name: str
    input_shape: str
    output_shape: str
    activation: str
    normalization: str
    encoder: str
    decoder: str
    temporal_block: str = "none"
    pooling: str = "none"
    initialization: str = "torch-default"


class ArchitectureRegistry:
    """Small shared registry for task-specific neural architecture builders."""

    def __init__(self):
        self._descriptors: dict[str, ArchitectureDescriptor] = {}
        self._builders: dict[str, Callable[..., object]] = {}

    def register(
        self,
        descriptor: ArchitectureDescriptor,
        builder: Callable[..., object],
    ) -> None:
        if descriptor.name in self._builders:
            raise ValueError(f"architecture {descriptor.name!r} is already registered")
        self._descriptors[descriptor.name] = descriptor
        self._builders[descriptor.name] = builder

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(sorted(self._builders))

    def descriptor(self, name: str) -> ArchitectureDescriptor:
        """Return metadata for one registered architecture."""

        try:
            return self._descriptors[name]
        except KeyError as exc:
            raise KeyError(f"unknown architecture {name!r}; choose from {self.names}") from exc

    def build(self, name: str, *args, **kwargs):
        """Build one registered architecture with task-specific arguments."""

        try:
            builder = self._builders[name]
        except KeyError as exc:
            raise KeyError(f"unknown architecture {name!r}; choose from {self.names}") from exc
        return builder(*args, **kwargs)

    def describe(
        self,
        name: str,
        components: nn.Module | Mapping[str, nn.Module],
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return descriptor, resolved configuration, and parameter counts."""

        if isinstance(components, nn.Module):
            component_counts = {"model": parameter_count(components)}
        else:
            component_counts = {
                str(component_name): parameter_count(component)
                for component_name, component in components.items()
            }
        return {
            **asdict(self.descriptor(name)),
            "configuration": dict(configuration or {}),
            "component_parameters": component_counts,
            "parameters": sum(component_counts.values()),
        }


def _activation(name: str) -> type[nn.Module]:
    choices: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(choices)}") from exc


class MLP(nn.Module):
    """Fully connected network with explicit width, depth, and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 2,
        activation: str | type[nn.Module] | Callable[[], nn.Module] = "tanh",
        final_activation: type[nn.Module] | Callable[[], nn.Module] | None = None,
    ):
        super().__init__()
        activation_factory = _activation(activation) if isinstance(activation, str) else activation
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, width), activation_factory()])
            current = width
        layers.append(nn.Linear(current, out_dim))
        if final_activation is not None:
            layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)


class CompartmentStateNet(nn.Module):
    """Time-to-state network normalized over ``compartments`` outputs."""

    def __init__(self, compartments: int = 3, width: int = 64, depth: int = 2):
        super().__init__()
        self.raw = MLP(1, compartments, width=width, depth=depth)

    def forward(self, time):
        return torch.softmax(self.raw(time), dim=-1)


class SimplexStateNet(CompartmentStateNet):
    """Three-compartment state network for SIR and SIPS examples."""

    def __init__(self, width: int = 64, depth: int = 2):
        super().__init__(compartments=3, width=width, depth=depth)


class BoundedControlNet(nn.Module):
    """Time-to-control network with values in ``[0, umax]``."""

    def __init__(self, width: int = 64, depth: int = 2, umax: float = 1.0):
        super().__init__()
        self.raw = MLP(1, 1, width=width, depth=depth)
        self.umax = float(umax)

    def forward(self, time):
        return self.umax * torch.sigmoid(self.raw(time))


class ResidualMLP(nn.Module):
    """Input/output projection around residual hidden blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        width: int = 64,
        depth: int = 3,
        activation: str = "tanh",
    ):
        super().__init__()
        act = _activation(activation)
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), act(), nn.Linear(width, width), act())
            for _ in range(depth)
        )
        self.output = nn.Linear(width, out_dim)

    def forward(self, values):
        hidden = self.input(values)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class FourierFeatures(nn.Module):
    """Deterministic sinusoidal embedding for scalar or vector coordinates."""

    def __init__(self, in_dim: int, features: int = 8, scale: float = 4.0):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        frequencies = torch.linspace(1.0, float(scale), features)
        self.register_buffer("frequencies", frequencies)
        self.in_dim = int(in_dim)

    @property
    def output_dim(self) -> int:
        return self.in_dim * self.frequencies.numel() * 2

    def forward(self, values):
        phase = 2.0 * math.pi * values.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)


class DenseGraphEncoder(nn.Module):
    """Small GCN-style encoder for dense normalized adjacency matrices.

    Input node features have shape ``[batch, nodes, features]``. The adjacency
    matrix may be ``[nodes, nodes]`` or ``[batch, nodes, nodes]``. The output has
    shape ``[batch, nodes, hidden]``.
    """

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, activation: str = "relu"):
        super().__init__()
        act = _activation(activation)
        widths = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(nn.Linear(widths[i], widths[i + 1]) for i in range(layers))
        self.activation = act()

    def forward(self, node_features, adjacency):
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, features]")
        if adjacency.ndim == 2:
            adjacency = adjacency.expand(node_features.shape[0], -1, -1)
        hidden = node_features
        for layer in self.layers:
            hidden = self.activation(layer(torch.bmm(adjacency, hidden)))
        return hidden


class FactorizedNodeTimeNet(nn.Module):
    """Factorized node-time state model with optional graph message passing.

    ``time`` has shape ``[batch, 1]`` and ``node_features`` has shape
    ``[nodes, node_dim]``. The returned SIPS state has shape
    ``[batch, nodes, compartments]`` and is normalized over compartments.
    """

    def __init__(
        self,
        node_dim: int,
        *,
        compartments: int = 3,
        width: int = 64,
        depth: int = 2,
        fourier_features: int = 8,
        graph_layers: int = 0,
    ):
        super().__init__()
        if fourier_features < 0:
            raise ValueError("fourier_features must be nonnegative")
        if fourier_features:
            self.time_features: nn.Module = FourierFeatures(1, fourier_features)
            time_dim = self.time_features.output_dim
        else:
            self.time_features = nn.Identity()
            time_dim = 1
        self.time_encoder = MLP(time_dim, width, width=width, depth=2)
        self.node_encoder = MLP(node_dim, width, width=width, depth=2)
        self.graph_encoder = (
            DenseGraphEncoder(width, hidden=width, layers=graph_layers) if graph_layers else None
        )
        self.decoder = MLP(2 * width, compartments, width=width, depth=depth)
        self.compartments = compartments

    def forward(self, time, node_features, adjacency=None):
        time_hidden = self.time_encoder(self.time_features(time))
        node_hidden = self.node_encoder(node_features)
        node_hidden = node_hidden.unsqueeze(0).expand(time.shape[0], -1, -1)
        if self.graph_encoder is not None:
            if adjacency is None:
                raise ValueError("adjacency is required when graph_layers > 0")
            node_hidden = self.graph_encoder(node_hidden, adjacency)
        time_hidden = time_hidden.unsqueeze(1).expand(-1, node_hidden.shape[1], -1)
        logits = self.decoder(torch.cat([time_hidden, node_hidden], dim=-1))
        return torch.softmax(logits, dim=-1)


def build_architecture(config: ArchitectureConfig) -> nn.Module:
    """Build an MLP or residual MLP from an inspectable configuration."""

    if config.encoder == "mlp":
        return MLP(
            config.input_dim,
            config.output_dim,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
        )
    if config.encoder == "residual":
        return ResidualMLP(
            config.input_dim,
            config.output_dim,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
        )
    raise ValueError(f"unsupported architecture encoder: {config.encoder}")


def parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Return the number of scalar parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_summary(model: nn.Module, config: ArchitectureConfig | None = None) -> dict[str, object]:
    """Return configuration and parameter count without printing."""

    return {
        "class": model.__class__.__name__,
        "parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "config": asdict(config) if config is not None else None,
    }


def shape_trace(model: nn.Module, *inputs) -> list[dict[str, object]]:
    """Run one forward pass and return leaf-module input/output shapes."""

    trace: list[dict[str, object]] = []
    handles = []

    def hook(name):
        def record(_module, values, output):
            in_shape = [tuple(value.shape) for value in values if hasattr(value, "shape")]
            if hasattr(output, "shape"):
                out_shape: object = tuple(output.shape)
            elif isinstance(output, (tuple, list)):
                out_shape = [tuple(value.shape) for value in output if hasattr(value, "shape")]
            else:
                out_shape = type(output).__name__
            trace.append({"layer": name, "input": in_shape, "output": out_shape})

        return record

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return trace
