import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.encoding import TemporalEncoding
from torch_geometric.nn.inits import zeros


class TemporalLinkInformation():
    def __init__(self, num_nodes: int, size: int, hidden_channels: int,
                 time_dim: int):
        super().__init__()

        self.num_nodes = num_nodes
        self.size = size
        self.hidden_channels = hidden_channels
        self.msg_store = torch.zeros((num_nodes, size, hidden_channels),
                                     dtype=torch.float)
        self.t = torch.zeros((num_nodes, size), dtype=torch.float)
        self.msg_count = torch.empty(num_nodes, dtype=torch.int)
        self.time_enc = TemporalEncoding(time_dim)
        self.reset_state()

    def __call__(self, n_id: Tensor, t_ref: Tensor) -> Tensor:
        msg = self.msg_store[n_id]
        t = self.t[n_id]
        msg_count = self.msg_count[n_id]
        mask = torch.arange(self.size).repeat(n_id.shape[0],
                                              1) < msg_count[:, None]
        encoded_t = self.time_enc(t_ref[:, None] - t)
        result = torch.cat([encoded_t, msg], dim=2) * mask[:, :, None]
        return result

    def insert(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor):
        for i in range(msg.shape[0]):
            self.msg_store[src[i]] = torch.cat(
                (msg[i][None, :], self.msg_store[src[i]][:-1]))
            self.msg_store[dst[i]] = torch.cat(
                (msg[i][None, :], self.msg_store[dst[i]][:-1]))
            self.t[src[i]] = torch.cat([t[i].reshape(1), self.t[src[i]][:-1]])
            self.t[dst[i]] = torch.cat([t[i].reshape(1), self.t[dst[i]][:-1]])
            self.msg_count[src[i]] += 1
            self.msg_count[dst[i]] += 1

    def reset_state(self):
        zeros(self.msg_count)
        zeros(self.t)


class MLPMixer(torch.nn.Module):
    """
    1-layer MLP Mixer
    """
    def __init__(self, dims, dropout=0):
        super().__init__()
        self.dims = dims
        self.dropout = dropout

        self.token_layernorm = torch.nn.LayerNorm(dims)
        self.token_linear = torch.nn.Linear(dims, dims)

        self.channel_layernorm = torch.nn.LayerNorm(dims)
        self.channel_linear = torch.nn.Linear(dims, dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.token_layernorm.reset_parameters()
        self.token_linear.reset_parameters()
        self.channel_layernorm.reset_parameters()
        self.channel_linear.reset_parameters()

    def mlp(self, linear, x):
        x = linear(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def token_mixer(self, x):
        x = self.token_layernorm(x)
        x = self.mlp(self.token_linear, x)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.mlp(self.channel_linear, x)
        return x

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class LinkEncoder(torch.nn.Module):
    def __init__(self, num_nodes: int, size: int, hidden_channels: int,
                 time_dim: int, dropout=0):
        super().__init__()
        self.temporal_link_information = TemporalLinkInformation(
            num_nodes=num_nodes, size=size, hidden_channels=hidden_channels,
            time_dim=time_dim)
        self.mlp_mixer = MLPMixer(dims=hidden_channels + time_dim,
                                  dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp_mixer.reset_parameters()

    def forward(self, n_id: Tensor, t_ref: Tensor):
        x = self.temporal_link_information(n_id, t_ref)
        x = self.mlp_mixer(x)
        x = torch.mean(x, dim=1)
        return x

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor):
        self.temporal_link_information.insert(src, dst, t, msg)

    def reset_state(self):
        self.temporal_link_information.reset_state()


class NodeEncoder:
    def __init__(self, num_nodes: int, memory_length: float,
                 node_emb: Tensor = None, device=None):
        super().__init__()
        if node_emb:
            self.node_emb = node_emb
        else:
            self.node_emb = torch.eye(num_nodes, device=device)
        self.node_dim = self.node_emb.shape[1]
        self.num_nodes = num_nodes
        self.memory_length = memory_length
        self.device = device
        self.reset_state()

    def __call__(self, n_id: Tensor, t: Tensor) -> Tensor:
        result = self.node_emb[n_id]
        for i in range(n_id.shape[0]):
            mask = self.neighbors_t[n_id[i]] > t[i] - self.memory_length
            self.neighbors_id[n_id[i]] = self.neighbors_id[n_id[i]][mask]
            self.neighbors_t[n_id[i]] = self.neighbors_t[n_id[i]][mask]
            if self.neighbors_id[n_id[i]].numel():
                result[i] = torch.mean(
                    self.node_emb[self.neighbors_id[n_id[i]]], dim=0)
        return result

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor):
        for i in range(src.shape[0]):
            self.neighbors_id[src[i]] = torch.cat(
                [self.neighbors_id[src[i]], dst[i].view(1)])
            self.neighbors_id[dst[i]] = torch.cat(
                [self.neighbors_id[dst[i]], src[i].view(1)])
            self.neighbors_t[src[i]] = torch.cat(
                [self.neighbors_t[src[i]], t[i].view(1)])
            self.neighbors_t[dst[i]] = torch.cat(
                [self.neighbors_t[dst[i]], t[i].view(1)])

    def reset_state(self):
        self.neighbors_id = [
            torch.empty(0, dtype=torch.long, device=self.device)
        ] * self.num_nodes
        self.neighbors_t = [
            torch.empty(0, dtype=torch.float, device=self.device)
        ] * self.num_nodes
