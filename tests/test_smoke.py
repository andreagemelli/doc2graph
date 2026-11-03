"""CPU smoke tests: no datasets, spaCy model or UNet needed."""

from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from doc2graph.data.graph_builder import GraphBuilder
from doc2graph.models.graphs import GcnSAGELayer, SetModel
from doc2graph.paths import CHECKPOINTS
from doc2graph.utils import set_preprocessing

NUM_POLAR_BINS = 8
CHUNKS = [4, 300, 1448]  # geometry, spaCy text, UNet visual (shipped checkpoints)


@pytest.fixture(scope="module", autouse=True)
def preprocessing_config():
    # get_model() reads configs/preprocessing.yaml, normally written by main.py
    set_preprocessing(
        Namespace(
            add_geom=True, add_embs=True, add_hist=False, add_visual=True,
            add_eweights=True, num_polar_bins=NUM_POLAR_BINS, src_data="FUNSD",
            data_type="img", edge_type="fully", node_granularity="gt",
        )
    )


def fully_connected_graph(n: int, node_feats: int) -> Data:
    u, v = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    e = len(u)
    return Data(
        edge_index=torch.tensor([u, v]),
        num_nodes=n,
        x=torch.randn(n, node_feats),
        edge_attr=torch.rand(e, 6),
        weights=torch.rand(e),
        norm=torch.rand(n, 1),
        y=torch.randint(0, 4, (n,)),
        edge_label=torch.randint(0, 2, (e,)),
    )


@pytest.mark.parametrize("name,weights", [("e2e", "e2e-funsd-best.pt"), ("edge", "edge-funsd-best.pt")])
def test_pretrained_checkpoint_loads_and_runs(name, weights):
    model = SetModel(name=name).get_model(4, 2, list(CHUNKS), False)
    model.load_state_dict(torch.load(CHECKPOINTS / weights, map_location="cpu"))
    g = Batch.from_data_list([fully_connected_graph(5, sum(CHUNKS)), fully_connected_graph(7, sum(CHUNKS))])
    out = model(g, g.x)
    e = out[1] if name == "e2e" else out
    assert e.shape == (g.num_edges, 2)
    if name == "e2e":
        assert out[0].shape == (g.num_nodes, 4)
    assert torch.isfinite(e).all()


def test_e2e_trains():
    model = SetModel(name="e2e").get_model(4, 2, [4, 8], False)
    g = Batch.from_data_list([fully_connected_graph(6, 12), fully_connected_graph(4, 12)])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        n, e = model(g, g.x)
        loss = torch.nn.functional.cross_entropy(n, g.y) + torch.nn.functional.cross_entropy(e, g.edge_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_message_passing_matches_naive_loop():
    g = fully_connected_graph(5, 3)
    layer = GcnSAGELayer(3, 4, None, 0.0, use_lynorm=False).eval()
    h = g.x
    agg = torch.zeros_like(h)
    for k, (s, d) in enumerate(g.edge_index.t().tolist()):
        agg[d] += h[s] * g.weights[k]
    expected = layer.linear(torch.cat((h, agg * g.norm), dim=1))
    assert torch.allclose(layer(g, h), expected, atol=1e-6)


def test_balance_edges_keeps_edge_attributes_aligned():
    g = fully_connected_graph(6, 3)
    g.edge_label = torch.zeros(g.num_edges, dtype=torch.long)
    g.edge_label[:4] = 1
    b = GraphBuilder.__new__(GraphBuilder).balance_edges(g, 0)
    assert int((b.edge_label == 1).sum()) == 4
    assert b.edge_index.shape[1] == b.edge_attr.shape[0] == b.weights.shape[0] == b.edge_label.shape[0]
    assert b.num_nodes == 6


def test_batch_roundtrip():
    graphs = [fully_connected_graph(3, 2), fully_connected_graph(4, 2)]
    batch = Batch.from_data_list(graphs)
    assert batch.num_graphs == 2
    assert [d.num_edges for d in batch.to_data_list()] == [6, 12]
