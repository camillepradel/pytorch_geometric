from typing import Union

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('add_node_feature')
class AddNodeFeature(BaseTransform):
    r"""Adds the node feature to :obj:`attr`
    (functional name: :obj:`add_node_feature`).

    Args:
        value (str, Tensor, optional): The value of the attribute to add.
        attr (str, optional): The attribute name of the data object to add
            features to. If set to :obj:`None`, will be
            concatenated or added to :obj:`data.x`.
            (default: :obj:`"x"`)
    """
    def __init__(self, value: Union[str, Tensor] = 'ones', attr: str = 'x'):
        if isinstance(value, str) and value not in {'zeros', 'ones', 'randn'}:
            raise ValueError(
                "`value` must be one of `'zeros'`, `'ones'`, "
                f"`'randn'` or an instance of `torch.Tensor`, but got {value}."
            )
        self.value = value
        self.attr = attr

    def __call__(self, data: Data) -> Data:
        value = self.value
        if isinstance(value, str):
            value = getattr(torch, value)(data.num_nodes, 1)

        x = data.get(self.attr)
        if x is not None:
            x = x.view(-1, 1) if x.dim() == 1 else x
            value = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        data[self.attr] = value
        return data
