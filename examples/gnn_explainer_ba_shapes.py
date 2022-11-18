import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import BAShapes
from torch_geometric.explain import Explainer, ExplainerConfig, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph

dataset = BAShapes(transform=T.GCNNorm())
data = dataset[0]

idx = torch.arange(data.num_nodes)
train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
            out_channels=dataset.num_classes, normalize=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model.forward(data.x, data.edge_index,
                         edge_weight=data.edge_weight).argmax(dim=-1)

    train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc


pbar = tqdm(range(1, 2001))
for epoch in pbar:
    loss = train()
    if epoch % 200 == 0:
        train_acc, test_acc = test()
        pbar.set_description(f'Loss: {loss:.4f}, '
                             f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
pbar.close()
model.eval()

for explanation_type in ["phenomenon", "model"]:
    targets, preds = [], []
    explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=300),
        explainer_config=ExplainerConfig(explanation_type=explanation_type,
                                         node_mask_type="attributes",
                                         edge_mask_type="object"),
        model_config=ModelConfig(mode="classification", task_level="node",
                                 return_type="raw"))

    # Explanation ROC AUC over all test nodes:
    loop_mask = data.edge_index[0] != data.edge_index[1]
    for node_idx in tqdm(
            data.expl_mask.nonzero(as_tuple=False).view(-1).tolist(),
            leave=False, desc="Training GNNExplainer"):
        explanation = explainer(x=data.x, edge_index=data.edge_index,
                                target_index=None, node_index=node_idx,
                                target=data.y, edge_weight=data.edge_weight)

        subgraph = k_hop_subgraph(node_idx, num_hops=3,
                                  edge_index=data.edge_index)
        expl_edge_mask = explanation.edge_mask[loop_mask]
        subgraph_edge_mask = subgraph[3][loop_mask]
        targets.append(data.edge_label[subgraph_edge_mask].cpu())
        preds.append(expl_edge_mask[subgraph_edge_mask].cpu())

    auc = roc_auc_score(torch.cat(targets), torch.cat(preds))
    print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}')
