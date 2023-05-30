import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import MLP
from torch_geometric.nn.models.graph_mixer import LinkEncoder, NodeEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(train_data, batch_size=200)
val_loader = TemporalDataLoader(val_data, batch_size=200)
test_loader = TemporalDataLoader(test_data, batch_size=200)

time_dim = 100
memory_length = 1000
link_encoder_size = 10
mlp_hidden_channels = 100

link_encoder = LinkEncoder(num_nodes=data.num_nodes, size=link_encoder_size,
                           hidden_channels=data.msg.size(-1),
                           time_dim=time_dim).to(device)
node_encoder = NodeEncoder(num_nodes=data.num_nodes,
                           memory_length=memory_length, device=device)
mlp = MLP([
    2 * (node_encoder.node_dim + time_dim + data.msg.size(-1)),
    mlp_hidden_channels, 1
])

optimizer = torch.optim.Adam(set(link_encoder.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()


# helper function to map global node indices to local ones within a batch
def _get_id_matching_node_and_time(node, time, n_id, t_ref):
    batch_size = node.shape[0]
    return ((n_id.repeat(batch_size, 1) == node.view(-1, 1).repeat(
        1, n_id.shape[0])) * (t_ref.repeat(batch_size, 1) == time.view(
            -1, 1).repeat(1, n_id.shape[0]))).nonzero()[:, 1]


def train():
    link_encoder.train()
    mlp.train()

    link_encoder.reset_state()
    node_encoder.reset_state()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pos_dst = batch.dst
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1,
                                (batch.src.size(0), ), dtype=torch.long,
                                device=device)

        n_id_t_ref_unique_pairs = torch.cat([
            torch.cat([batch.src, pos_dst, neg_dst])[None, :],
            batch.t.repeat(3)[None, :]
        ], dim=0).unique(dim=1)
        n_id = n_id_t_ref_unique_pairs[0]
        t_ref = n_id_t_ref_unique_pairs[1]

        t = link_encoder(n_id, t_ref)
        s = node_encoder(n_id, t_ref)
        h = torch.cat([s, t], dim=1)

        src_id = _get_id_matching_node_and_time(batch.src, batch.t, n_id,
                                                t_ref)
        pos_dst_id = _get_id_matching_node_and_time(pos_dst, batch.t, n_id,
                                                    t_ref)
        neg_dst_id = _get_id_matching_node_and_time(neg_dst, batch.t, n_id,
                                                    t_ref)

        pos_h = torch.cat([h[src_id], h[pos_dst_id]], dim=1)
        neg_h = torch.cat([h[src_id], h[neg_dst_id]], dim=1)

        pos_out = mlp(pos_h)
        neg_out = mlp(neg_h)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        link_encoder.update_state(batch.src, batch.dst, batch.t, batch.msg)
        node_encoder.update_state(batch.src, batch.dst, batch.t)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    link_encoder.train()
    mlp.train()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)

        pos_dst = batch.dst
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1,
                                (batch.src.size(0), ), dtype=torch.long,
                                device=device)

        n_id_t_ref_unique_pairs = torch.cat([
            torch.cat([batch.src, pos_dst, neg_dst])[None, :],
            batch.t.repeat(3)[None, :]
        ], dim=0).unique(dim=1)
        n_id = n_id_t_ref_unique_pairs[0]
        t_ref = n_id_t_ref_unique_pairs[1]

        l_enc = link_encoder(n_id, t_ref)
        n_enc = node_encoder(n_id, t_ref)
        h = torch.cat([n_enc, l_enc], dim=1)

        src_id = _get_id_matching_node_and_time(batch.src, batch.t, n_id,
                                                t_ref)
        pos_dst_id = _get_id_matching_node_and_time(pos_dst, batch.t, n_id,
                                                    t_ref)
        neg_dst_id = _get_id_matching_node_and_time(neg_dst, batch.t, n_id,
                                                    t_ref)

        pos_h = torch.cat([h[src_id], h[pos_dst_id]], dim=1)
        neg_h = torch.cat([h[src_id], h[neg_dst_id]], dim=1)

        pos_out = mlp(pos_h)
        neg_out = mlp(neg_h)

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        link_encoder.update_state(batch.src, batch.dst, batch.t, batch.msg)
        node_encoder.update_state(batch.src, batch.dst, batch.t)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_loader)
    test_ap, test_auc = test(test_loader)
    print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
