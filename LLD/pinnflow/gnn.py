"""
Graph neural network helpers for topology analysis.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx


class GasNetworkGNN(torch.nn.Module):
    """
    GNN for gas network flow prediction.
    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 64, graph_embedding_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.graph_head = torch.nn.Linear(hidden_dim, graph_embedding_dim)
        self.press_head = torch.nn.Linear(hidden_dim, 1)
        self.flow_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return {
            "node_pressures": self.press_head(h),
            "flows": self.flow_head(h),
            "graph_embedding": self.graph_head(global_mean_pool(h, batch)),
        }


def load_gaslib_graph(topology_name: str, mode: str | None = None):
    """
    Load a scenario-selected topology and attach deterministic structural features.
    """
    topology_key = str(topology_name).lower()

    if mode is not None:
        selected_mode = mode
    elif "synthetic" in topology_key or "fsi" in topology_key:
        selected_mode = "synthetic"
    else:
        selected_mode = "authentic"

    if topology_key in {"gaslib-11", "gaslib11"}:
        name = "GasLib-11"
    elif topology_key in {"gaslib-24", "gaslib24"}:
        name = "GasLib-24"
    elif topology_key in {"gaslib-39", "gaslib39"}:
        name = "GasLib-39"
    elif topology_key in {"gaslib-40", "gaslib40"}:
        name = "GasLib-40"
    elif topology_key in {"gaslib-134", "gaslib134"}:
        name = "GasLib-134"
    elif topology_key in {"gaslib-582", "gaslib582"}:
        name = "GasLib-582"
    elif topology_key in {"gaslib-4197", "gaslib4197"}:
        name = "GasLib-4197"
    elif topology_key in {"gaslib-2607", "gaslib2607"}:
        name = "GasLib-2607"
    else:
        name = topology_name

    try:
        from data.gaslib_loader import GasLibLoader

        graph = GasLibLoader().load_network(name, mode=selected_mode)
    except ModuleNotFoundError:
        graph = _fallback_graph(name)

    nodes = list(graph.nodes())
    node_count = max(len(nodes), 1)
    x = torch.zeros((node_count, 8), dtype=torch.float32)
    for idx, node in enumerate(nodes):
        out_degree = graph.out_degree(node)
        in_degree = graph.in_degree(node)
        node_attrs = graph.nodes[node]
        node_kind = str(node_attrs.get("kind", node_attrs.get("type", "junction"))).lower()
        x[idx, 0] = float(out_degree + in_degree)
        x[idx, 1] = 1.0 if node_kind == "source" else 0.0
        x[idx, 2] = 1.0 if node_kind == "sink" else 0.0
        x[idx, 3] = idx / max(node_count - 1, 1)
        component_id = 0
        if "elbow" in node_kind or "bend" in node_kind:
            component_id = 1
        elif "tee" in node_kind or "junction" in node_kind:
            component_id = 2
        elif "reducer" in node_kind:
            component_id = 3
        x[idx, 4 + component_id] = 1.0

    edge_pairs = []
    edge_attr_rows = []
    for src, dst, attrs in graph.edges(data=True):
        edge_pairs.append([nodes.index(src), nodes.index(dst)])
        length = float(attrs.get("length", attrs.get("design_length_m", 1.0)) or 1.0)
        diameter = float(attrs.get("diameter", 273.0) or 273.0)
        shape_id = float(attrs.get("shape_id", 0.0) or 0.0)
        r_over_d = float(attrs.get("shape_param", attrs.get("R_D_ratio", 1.0)) or 1.0)
        edge_attr_rows.append([length, diameter, shape_id, r_over_d])
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32) if edge_attr_rows else torch.empty((0, 4), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = node_count
    data.x = x
    data.topology_name = topology_name
    data.topology_mode = selected_mode
    data.node_names = nodes
    return data


def _fallback_graph(name: str):
    """Build a deterministic topology when the optional GasLib loader is absent."""
    graph = nx.DiGraph()
    graph.add_node("source", kind="source")
    graph.add_node("bend", kind="elbow")
    graph.add_node("junction", kind="junction")
    graph.add_node("sink_a", kind="sink")
    graph.add_node("sink_b", kind="sink")
    graph.add_edge("source", "bend", length=75.0, diameter=323.9, shape_id=0.0, shape_param=1.0)
    graph.add_edge("bend", "junction", length=45.0, diameter=323.9, shape_id=1.0, shape_param=3.0)
    graph.add_edge("junction", "sink_a", length=60.0, diameter=273.0, shape_id=2.0, shape_param=1.0)
    graph.add_edge("junction", "sink_b", length=55.0, diameter=219.1, shape_id=3.0, shape_param=1.4)
    graph.graph["name"] = name
    graph.graph["fallback"] = True
    return graph
