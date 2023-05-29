import torch

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.encoding import TemporalEncoding
from torch_geometric.nn.models.graph_mixer import (
    LinkEncoder,
    TemporalLinkInformation,
)


def test_temporal_link_information():
    hidden_channels = 16
    time_dim = 16

    src = torch.tensor([0, 1, 0, 2, 0, 3, 1, 4, 2, 3])
    dst = torch.tensor([1, 2, 1, 1, 3, 2, 4, 3, 3, 4])
    t = torch.arange(10)
    msg = torch.randn(10, hidden_channels)
    data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    loader = TemporalDataLoader(data, batch_size=5)

    temporal_link_information = TemporalLinkInformation(
        num_nodes=data.num_nodes, size=3, hidden_channels=hidden_channels,
        time_dim=time_dim)
    assert temporal_link_information.num_nodes == data.num_nodes
    assert temporal_link_information.size == 3
    assert temporal_link_information.hidden_channels == hidden_channels
    assert temporal_link_information.msg_store.size() == (data.num_nodes, 3,
                                                          hidden_channels)
    assert temporal_link_information.t.size() == (data.num_nodes, 3)
    assert temporal_link_information.msg_count.size() == (data.num_nodes, )

    # running twice below code allow to test reset_state()
    for _ in range(2):
        for i, batch in enumerate(loader):
            n_id_t_ref_unique_pairs = torch.cat([
                torch.cat([batch.src, batch.dst])[None, :],
                batch.t.repeat(2)[None, :]
            ], dim=0).unique(dim=1)
            n_id = n_id_t_ref_unique_pairs[0]
            t_ref = n_id_t_ref_unique_pairs[1]
            tli = temporal_link_information(n_id, t_ref)
            temporal_link_information.insert(batch.src, batch.dst, batch.t,
                                             batch.msg)
            assert tli.size(0) == n_id.size(0) == t.size(0)
            assert tli.size(1) == 3
            assert tli.size(2) == hidden_channels + time_dim
            if i == 0:
                assert n_id.size(0) == 10
                assert torch.equal(
                    tli,
                    torch.zeros([n_id.size(0), 3, hidden_channels + time_dim]))
            else:
                # in below dict, keys are node ids and values are lists in
                # which each item index refers to the index in the temporal
                # link information memory and value to the msg index and time
                # value
                n_id_to_expected_msg = {
                    3: [4],
                    2: [3, 1],
                    1: [3, 2, 1],
                    4: [],
                }
                time_enc = TemporalEncoding(time_dim)
                for i in range((tli.size(0))):
                    expected_msg = n_id_to_expected_msg[n_id[i].item()]
                    for history_index, msg_index in enumerate(expected_msg):
                        t_value = msg_index
                        assert torch.equal(
                            tli[i, history_index, :time_dim],
                            time_enc(torch.Tensor([t_ref[i] - t_value]))[0])
                        assert torch.equal(tli[i, history_index, time_dim:],
                                           msg[msg_index])
                    # also check spots which are supposed to be empty
                    for history_index in range(len(expected_msg), 3):
                        assert torch.equal(
                            tli[i, history_index],
                            torch.zeros([hidden_channels + time_dim]))
        temporal_link_information.reset_state()


def test_link_encoder():
    hidden_channels = 16
    time_dim = 16

    src = torch.tensor([0, 1, 0, 2, 0, 3, 1, 4, 2, 3])
    dst = torch.tensor([1, 2, 1, 1, 3, 2, 4, 3, 3, 4])
    t = torch.arange(10)
    msg = torch.randn(10, hidden_channels)
    data = TemporalData(src=src, dst=dst, t=t, msg=msg)

    loader = TemporalDataLoader(data, batch_size=5)

    link_encoder = LinkEncoder(num_nodes=data.num_nodes, size=3,
                               hidden_channels=hidden_channels,
                               time_dim=time_dim)

    for i, batch in enumerate(loader):
        n_id_t_ref_unique_pairs = torch.cat([
            torch.cat([batch.src, batch.dst])[None, :],
            batch.t.repeat(2)[None, :]
        ], dim=0).unique(dim=1)
        n_id = n_id_t_ref_unique_pairs[0]
        t_ref = n_id_t_ref_unique_pairs[1]
        z = link_encoder(n_id, t_ref)
        link_encoder.update_state(batch.src, batch.dst, batch.t, batch.msg)

        assert z.size(0) == n_id.size(0)
        assert z.size(1) == hidden_channels + time_dim


if __name__ == "__main__":
    # test_temporal_link_information()
    test_link_encoder()
