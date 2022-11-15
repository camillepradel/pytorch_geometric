import logging
from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.explain.explanations import Explanation

from .base import ExplainerAlgorithm


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN's node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        shared_mask (bool, optional): Whether to share the mask for node
            features. When set to :obj:`True`, the node feature mask will be
            shared across all nodes. (default: :obj:`True`). Only used when the
            node mask type is :obj:`MaskType.attributes`.
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.node_mask = None
        self.edge_mask = None

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        node_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:

        if isinstance(target_index, Tensor) and target_index.numel() > 1:
            raise ValueError(
                "GNNExplainer only supports single target index for now")

        assert model_config.task_level in [
            ModelTaskLevel.graph, ModelTaskLevel.node
        ]

        model.eval()

        if model_config.task_level == ModelTaskLevel.node:
            node_mask, edge_mask = self._explain_node(
                model,
                x,
                edge_index,
                explainer_config,
                model_config,
                target,
                node_index,
                target_index,
                **kwargs,
            )
        elif model_config.task_level == ModelTaskLevel.graph:
            node_mask, edge_mask = self._explain_graph(
                model,
                x,
                edge_index,
                explainer_config,
                model_config,
                target,
                target_index,
                **kwargs,
            )
        if explainer_config.node_mask_type == MaskType.attributes:
            if explainer_config.node_mask_type == MaskType.common_attributes:
                node_feat_mask = torch.stack([node_mask] * x.size(0), dim=0)
            else:
                node_feat_mask = node_mask
            node_mask = None
        else:
            node_feat_mask = None
        self._clean_model(model)

        # build explanation
        return Explanation(x=x, edge_index=edge_index, edge_mask=edge_mask,
                           node_mask=node_mask, node_feat_mask=node_feat_mask)

    def _explain_graph(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        self.train_node_edge_mask(
            model,
            x,
            edge_index,
            explainer_config,
            model_config,
            target,
            target_index,
            None,
            kwargs,
        )

        node_mask = self.node_mask.detach().sigmoid().squeeze(-1)
        if explainer_config.edge_mask_type == MaskType.object:
            edge_mask = self.edge_mask.detach().sigmoid()
        else:
            edge_mask = None
        return node_mask, edge_mask

    def _explain_node(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        index: Optional[Union[int, Tensor]],
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if index is None:
            raise ValueError("Please provide node index for node-level "
                             "explanation")

        # if we are dealing with a node level task, we can restrict the
        # computation to the node of interest and its computation graph
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        x, edge_index, index, subset, hard_edge_mask, kwargs =\
            self.subgraph(model, index, x, edge_index, **kwargs)
        if target_index is not None and model_config.mode ==\
                ModelMode.classification:
            target = torch.index_select(target, 1, subset)
        else:
            target = target[subset]

        self.train_node_edge_mask(
            model,
            x,
            edge_index,
            explainer_config,
            model_config,
            target,
            target_index,
            index,
            kwargs,
        )

        if explainer_config.node_mask_type == MaskType.common_attributes:
            node_mask = self.node_mask.detach().sigmoid().squeeze(-1)
        else:
            if explainer_config.node_mask_type == MaskType.object:
                new_mask = x.new_zeros(num_nodes, 1)
            if explainer_config.node_mask_type == MaskType.attributes:
                new_mask = x.new_zeros(num_nodes, x.size(-1))
            new_mask[subset] = self.node_mask.detach().sigmoid()
            node_mask = new_mask.squeeze(-1)

        if explainer_config.edge_mask_type == MaskType.object:
            new_edge_mask = torch.zeros(num_edges)
            new_edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()
            edge_mask = new_edge_mask
        else:
            edge_mask = None

        return node_mask, edge_mask

    def train_node_edge_mask(
        self,
        model,
        x,
        edge_index,
        explainer_config,
        model_config,
        target,
        target_index,
        index,
        kwargs,
    ):
        self._initialize_masks(x, edge_index,
                               node_mask_type=explainer_config.node_mask_type,
                               edge_mask_type=explainer_config.edge_mask_type)

        if explainer_config.edge_mask_type == MaskType.object:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters = [self.node_mask, self.edge_mask]
        else:
            parameters = [self.node_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_mask.sigmoid()
            out = model(x=h, edge_index=edge_index, **kwargs)
            loss_value = self.loss(
                out, target, return_type=model_config.return_type,
                target_idx=target_index, node_index=index,
                edge_mask_type=explainer_config.edge_mask_type,
                model_mode=model_config.mode)
            loss_value.backward(retain_graph=True)
            optimizer.step()

    def _initialize_masks(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_mask_type: MaskType,
        edge_mask_type: MaskType,
    ):
        (N, F), E = x.size(), edge_index.size(1)
        device = x.device
        std = 0.1
        if node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        else:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)

        if edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)

    def loss_regression(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        target_idx: Optional[int] = None,
        node_index: Optional[int] = None,
    ):
        if target_idx is not None:
            y_hat = y_hat[..., target_idx].unsqueeze(-1)
            y = y[..., target_idx].unsqueeze(-1)

        if node_index is not None and node_index >= 0:
            loss_ = torch.cdist(y_hat[node_index], y[node_index])
        else:
            loss_ = torch.cdist(y_hat, y)

        return loss_

    def loss_classification(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        return_type: ModelReturnType,
        target_idx: Optional[int] = None,
        node_index: Optional[int] = None,
    ):
        if target_idx is not None:
            y_hat = y_hat[target_idx]
            y = y[target_idx]

        y_hat = self._to_log_prob(y_hat, return_type)

        if node_index is not None and node_index >= 0:
            loss = -y_hat[node_index, y[node_index]]
        else:
            loss = -y_hat[0, y[0]]
        return loss

    def loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        edge_mask_type: MaskType,
        return_type: ModelReturnType,
        node_index: Optional[int] = None,
        target_idx: Optional[int] = None,
        model_mode: ModelMode = ModelMode.regression,
    ) -> torch.Tensor:

        if model_mode == ModelMode.regression:
            loss = self.loss_regression(y_hat, y, target_idx, node_index)
        else:
            loss = self.loss_classification(y_hat, y, return_type, target_idx,
                                            node_index)

        if edge_mask_type is not None:
            m = self.edge_mask.sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def supports(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:
        if model_config.task_level not in [
                ModelTaskLevel.node, ModelTaskLevel.graph
        ]:
            logging.error("Model task level not supported.")
            return False
        if explainer_config.edge_mask_type == MaskType.attributes:
            logging.error("Edge mask type not supported.")
            return False

        if explainer_config.node_mask_type is None:
            logging.error("Node mask type not supported.")
            return False

        if model_config.task_level not in [
                ModelTaskLevel.node, ModelTaskLevel.graph
        ]:
            logging.error("Model task level not supported.")
            return False

        return True

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = None
        self.edge_mask = None
