# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

"""Minimal heterogeneous degree-class optimal-control example.

The script loads a graph, forms its empirical degree classes, calls the
canonical :mod:`cybercontrol.fbs` solver, and writes trajectories and numerical
diagnostics. It contains no alternate model or forward-backward sweep.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cybercontrol.baseline_diagnostics import RANDOM_BASELINE_COUNT
from cybercontrol.fbs import degree_control_model, solve_degree_control
from cybercontrol.foundation_profiles import DEGREE_CONTROL_PROFILE
from cybercontrol.graph_data import (
    DegreeData,
    degree_distribution_from_graph,
    load_network,
    print_degree_distribution,
)

try:
    from .foundation_reporting import (
        plot_degree_control,
        save_baseline_comparison,
        save_degree_distribution,
        save_fbs_convergence,
    )
except ImportError:
    from foundation_reporting import (
        plot_degree_control,
        save_baseline_comparison,
        save_degree_distribution,
        save_fbs_convergence,
    )


LOGGER = logging.getLogger(__name__)


def write_resolved_parameters(
    out_dir: Path,
    degree_data: DegreeData,
    *,
    steps: int,
    max_iterations: int,
    tolerance: float,
) -> None:
    """Write the exact heterogeneous arrays and a compact run summary."""

    model = degree_control_model(degree_data)
    pd.DataFrame(
        {
            "degree_k": degree_data.k,
            "class_count": degree_data.Nk,
            "class_probability": degree_data.pk,
            "susceptibility": model.susceptibility,
            "infectivity": model.infectivity,
            "recovery": model.recovery,
            "state_weight": model.state_weight,
            "control_weight": model.control_weight,
            "control_bound": model.control_bound,
        }
    ).to_csv(out_dir / "resolved_degree_parameters.csv", index=False)
    pd.DataFrame(model.mixing, index=degree_data.k, columns=degree_data.k).to_csv(
        out_dir / "degree_mixing_matrix.csv",
        index_label="degree_k",
    )

    rows: list[dict[str, object]] = [
        {
            "parameter": "model_level",
            "value": "degree class",
            "meaning": "Each state and control entry corresponds to one observed degree k.",
        },
        {
            "parameter": "heterogeneity_profile",
            "value": f"degree-correlated, strength={DEGREE_CONTROL_PROFILE.heterogeneity_strength}",
            "meaning": "Physical and economic coefficients vary deterministically by degree class.",
        },
        {
            "parameter": "horizon",
            "value": DEGREE_CONTROL_PROFILE.horizon,
            "meaning": "Total simulated time.",
        },
        {
            "parameter": "time_grid_steps",
            "value": steps,
            "meaning": "Number of time intervals in the ODE and FBS grids.",
        },
        {
            "parameter": "max_fbs_iterations",
            "value": max_iterations,
            "meaning": "Maximum forward-backward sweep updates.",
        },
        {
            "parameter": "fbs_tolerance",
            "value": tolerance,
            "meaning": "Stopping threshold for the maximum control change.",
        },
        {
            "parameter": "random_baselines",
            "value": RANDOM_BASELINE_COUNT,
            "meaning": "Number of seeded random smooth controls in the comparison.",
        },
    ]
    for summary in model.summaries():
        name = str(summary["name"])
        rows.append(
            {
                "parameter": f"{name}_range",
                "value": f"{summary['min']:.6g} to {summary['max']:.6g}",
                "meaning": f"Resolved class-wise {name}; mean={summary['mean']:.6g}.",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "parameter_summary.csv", index=False)


def run(args: argparse.Namespace) -> None:
    """Execute the minimal graph-to-control workflow."""

    network = load_network(args)
    degree_data = degree_distribution_from_graph(network.full_graph, mode="undirected")
    result = solve_degree_control(
        degree_data,
        steps=args.steps,
        iterations=args.iterations,
        tol=args.tolerance,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Loaded %s: n=%d, m=%d",
        network.source,
        network.full_graph.number_of_nodes(),
        network.full_graph.number_of_edges(),
    )
    print_degree_distribution(degree_data)
    save_degree_distribution(degree_data, out_dir)
    plot_degree_control(result, degree_data, out_dir)
    save_fbs_convergence({"degree_control": result}, out_dir)
    save_baseline_comparison({"degree_control": result}, degree_data, network.A, out_dir)
    write_resolved_parameters(
        out_dir,
        degree_data,
        steps=args.steps,
        max_iterations=args.iterations,
        tolerance=args.tolerance,
    )
    LOGGER.info(
        "FBS converged=%s after %d iterations; cost=%.6f; outputs=%s",
        bool(result.value["converged"]),
        int(result.value["iterations"]),
        float(result.value["cost"]),
        out_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the graph-input and bounded-solver command-line options."""

    parser = argparse.ArgumentParser(
        description="Minimal heterogeneous degree-class optimal-control example."
    )
    parser.add_argument("--edge-list", type=str, help="Source-target edge list.")
    parser.add_argument("--adjacency-csv", type=str, help="Square adjacency matrix CSV.")
    parser.add_argument("--graph-file", type=str, help="GraphML, GEXF, GML, or Pajek file.")
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--source-col", default=0)
    parser.add_argument("--target-col", default=1)
    parser.add_argument("--weight-col", default=None)
    parser.add_argument("--demo-nodes", type=int, default=30)
    parser.add_argument("--demo-m", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/simple_degree_control"))
    parser.set_defaults(
        adjacency_convention="graph",
        directed=False,
        graph_backend="networkx",
        max_nodes=0,
        normalize="none",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
