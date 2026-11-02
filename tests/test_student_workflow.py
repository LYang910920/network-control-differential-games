"""Student-facing input and end-to-end example tests."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from cybercontrol.cli import build_parser, run_example
from cybercontrol.graph_data import (
    degree_distribution_from_graph,
    edge_list_to_graph_igraph,
    graph_to_model_matrix,
    load_graph,
    load_network,
    prepare_model_matrix,
)


@pytest.fixture
def edge_args(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("source,target,weight\n001,002,2\n002,001,3\n003,003,1\n")
    return SimpleNamespace(
        edge_list=str(path),
        delimiter=",",
        has_header=True,
        source_col="source",
        target_col="target",
        weight_col="weight",
        directed=False,
        adjacency_csv=None,
        adjacency_convention="graph",
        graph_file=None,
        graph_backend="networkx",
        max_nodes=2,
        normalize="none",
    )


def test_labels_duplicate_edges_and_isolates(edge_args):
    graph, _ = load_graph(edge_args)
    degree = degree_distribution_from_graph(graph)
    assert set(graph) == {"001", "002", "003"}
    assert graph["001"]["002"]["weight"] == 5
    assert np.array_equal(degree.k, [0, 1])
    assert np.array_equal(degree.Nk, [1, 2])
    assert np.isclose(degree.kbar, 2 / 3)


def test_degree_loading_does_not_allocate_a_dense_matrix(edge_args, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Degree statistics must not allocate a dense adjacency matrix.")

    monkeypatch.setattr(nx, "to_numpy_array", forbidden)
    graph, _ = load_graph(edge_args)
    assert degree_distribution_from_graph(graph).Nk.sum() == 3


def test_node_reduction_does_not_change_full_degree_statistics(edge_args):
    network = load_network(edge_args)
    assert network.A.shape == (2, 2)
    assert degree_distribution_from_graph(network.full_graph).Nk.sum() == 3


def test_igraph_and_networkx_have_identical_edge_semantics(edge_args):
    pytest.importorskip("igraph")
    graph, _ = load_graph(edge_args)
    other = edge_list_to_graph_igraph(edge_args)
    assert set(graph) == set(other)
    assert other["001"]["002"]["weight"] == graph["001"]["002"]["weight"]
    assert np.array_equal(
        degree_distribution_from_graph(graph).Nk, degree_distribution_from_graph(other).Nk
    )


def test_directed_adjacency_uses_incoming_influence(edge_args):
    edge_args.directed = True
    graph, _ = load_graph(edge_args)
    matrix, labels = graph_to_model_matrix(graph, directed=True, normalize="none")
    assert matrix[labels.index("002"), labels.index("001")] == 2
    assert matrix[labels.index("001"), labels.index("002")] == 3


def test_invalid_rates_are_not_silently_changed():
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_model_matrix(np.array([[0.0, -1.0], [0.0, 0.0]]), "none")
    with pytest.raises(ValueError, match="finite"):
        prepare_model_matrix(np.array([[0.0, np.nan], [0.0, 0.0]]), "none")


def test_first_example_writes_only_learning_outputs(tmp_path):
    args = build_parser().parse_args(
        ["--demo-nodes", "12", "--steps", "12", "--iterations", "60"]
    )
    args.output_dir = tmp_path
    comparison = run_example(args)
    assert set(path.name for path in tmp_path.iterdir()) == {
        "degree_distribution.csv",
        "trajectories.csv",
        "comparison.csv",
        "infection.png",
        "control.png",
    }
    assert np.isfinite(comparison[["cost", "infected_time"]].to_numpy()).all()
    values = comparison.set_index("policy")["cost"]
    assert values["PMP candidate"] <= values["No control"]
