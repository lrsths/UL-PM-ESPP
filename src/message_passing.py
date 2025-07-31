import torch
import torch.nn.functional as F
from torch.nn import GRUCell, Linear, Sequential, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_logsumexp
from torch_geometric.utils import softmax


class MultiHopEdgeGATConv_ordering(MessagePassing):
    def __init__(self, in_channels, edge_in_dim, *, K=3, edge_hidden_dim=128,
                 pos_dim=2, negative_slope=0.2, dropout=0.25,
                 valid_hops=(1, 2, 3), checkpoint=False, topological_mask=None, spec_dim=3):
        super().__init__(aggr='add', node_dim=0)

        self.K = K
        self.in_channels = in_channels
        self.concat_dim = 2 * in_channels
        self.edge_in_dim = edge_in_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.valid_hops = valid_hops
        self.checkpoint = checkpoint
        self.topological_mask = topological_mask
        self.spec_dim = spec_dim
        self.lin_src = Linear(self.concat_dim, in_channels, bias=False)
        self.lin_dst = Linear(self.concat_dim, in_channels, bias=False)
        self.gru = GRUCell(in_channels, in_channels)

        att_in_dim = 2 * in_channels + edge_in_dim + pos_dim

        self.att_mlp = Sequential(
            Linear(att_in_dim, edge_hidden_dim), ReLU(),
            Linear(edge_hidden_dim, edge_hidden_dim // 2), ReLU(),
            Linear(edge_hidden_dim // 2, 1)
        )
        self.edge_mlp = Sequential(
            Linear(2 * in_channels + edge_in_dim, edge_hidden_dim), ReLU(),
            Linear(edge_hidden_dim, edge_in_dim)
        )

        self.edge_proj = Linear(edge_in_dim, 1)

        self.edge_emb = Sequential(
            Linear(edge_in_dim, in_channels), ReLU(),
            Linear(in_channels, in_channels)
        )

        self.gate_mlp = Sequential(Linear(edge_in_dim, in_channels), Sigmoid())

        self.node_score_mlp = Sequential(
            Linear(in_channels, in_channels), ReLU(),
            Linear(in_channels, 1)
        )

        self.out_val = Sequential(
            Linear(in_channels, in_channels), ReLU(),
            Linear(in_channels, 1)
        )

        self.node_msg_mlp_i = Sequential(
            Linear(self.in_channels + self.edge_in_dim + spec_dim, self.in_channels * 2), ReLU(),
            Linear(self.in_channels * 2, self.in_channels)
        )

        self.node_msg_mlp_j = Sequential(
            Linear(self.in_channels + self.edge_in_dim + spec_dim, self.in_channels * 4), ReLU(),
            Linear(self.in_channels * 4, self.in_channels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.gru.reset_parameters()
        for layer in (self.att_mlp, self.edge_proj, self.edge_mlp,
                      self.edge_emb, self.gate_mlp, self.node_score_mlp, self.out_val):
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _one_step(self, x, h_prev, pos, edge_index, edge_attr, hop_count, spec):
        x_src = h_prev[edge_index[0]]
        x_dst = h_prev[edge_index[1]]
        edge_input = torch.cat([x_src, edge_attr, x_dst], dim=-1)
        edge_attr = self.edge_mlp(edge_input)

        h_input = torch.cat([x, h_prev], dim=-1)
        h_src = self.lin_src(h_input)
        h_dst = self.lin_dst(h_input)

        self.edge_index = edge_index
        m = self.propagate(edge_index=edge_index, x=(h_src, h_dst), pos=pos,
                           edge_attr=edge_attr, hop_count=hop_count, spec=spec)
        m = torch.zeros_like(h_src) if m is None else m
        new_h = torch.min(m, h_prev)

        return new_h, edge_attr, hop_count + 1

    def message(self, x_i, x_j, spec_i, spec_j, edge_attr, index, hop_count):
        delta_spec = spec_j - spec_i
        x_i_input = torch.cat([x_j, edge_attr, spec_i], dim=-1)
        x_j_input = torch.cat([x_j - x_i, edge_attr, delta_spec], dim=-1)

        x_i_proj = self.node_msg_mlp_i(x_i_input)
        x_j_proj = self.node_msg_mlp_j(x_j_input)

        edge_cost = self.edge_proj(edge_attr).squeeze(-1)  # shape [E]
        alpha = softmax(-edge_cost, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return (x_j_proj) * alpha.unsqueeze(-1)

    def forward(self, x, pos, edge_index, edge_attr, spec, node_scores=None,
                topological_mask=None):  # *,teacher_rev, teacher_fwd):
        self.node_scores = self.node_score_mlp(x).squeeze(-1)

        h = torch.zeros_like(x)
        hop_count = torch.zeros(edge_index.size(1), 1, device=x.device)

        for _ in range(self.K):
            h, edge_attr, hop_count = self._one_step(x, h, pos, edge_index, edge_attr, hop_count, spec)

        node_values = self.out_val(h).squeeze(-1)

        return h, edge_attr, self.node_scores, node_values