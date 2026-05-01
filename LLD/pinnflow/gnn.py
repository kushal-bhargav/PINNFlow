"""
Graph neural network helpers for topology analysis.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GasNetworkGNN(torch.nn.Module):
    """
    GNN for gas network flow prediction.
    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.press_head = torch.nn.Linear(hidden_dim, 1)
        self.flow_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return {
            "node_pressures": self.press_head(h),
            "flows": self.flow_head(h),
        }


def load_gaslib_graph(topology_name: str, mode: str | None = None):
    """
    Load a scenario-selected topology and attach deterministic structural features.
    """
    from data.gaslib_loader import GasLibLoader

    loader = GasLibLoader()
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

    graph = loader.load_network(name, mode=selected_mode)

    nodes = list(graph.nodes())
    node_count = max(len(nodes), 1)
    x = torch.zeros((node_count, 4), dtype=torch.float32)
    for idx, node in enumerate(nodes):
        out_degree = graph.out_degree(node)
        in_degree = graph.in_degree(node)
        node_attrs = graph.nodes[node]
        node_kind = str(node_attrs.get("kind", node_attrs.get("type", "junction"))).lower()
        x[idx, 0] = float(out_degree + in_degree)
        x[idx, 1] = 1.0 if node_kind == "source" else 0.0
        x[idx, 2] = 1.0 if node_kind == "sink" else 0.0
        x[idx, 3] = idx / max(node_count - 1, 1)

    edge_pairs = []
    for src, dst in graph.edges():
        edge_pairs.append([nodes.index(src), nodes.index(dst)])
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = node_count
    data.x = x
    data.topology_name = topology_name
    data.topology_mode = selected_mode
    data.node_names = nodes
    return data
