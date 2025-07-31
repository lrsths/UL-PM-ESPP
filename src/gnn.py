import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import BatchNorm as BN
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_add, scatter_logsumexp, scatter_softmax, scatter_mean, scatter_min, \
    scatter_max
from src.message_passing import MultiHopEdgeGATConv_ordering


class ShortestPathGNN_edge_hop(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hidden1, hidden2, *,
                 edge_in_dim=2, edge_hidden_dim=128, pos_dim=2,
                 dropout=0.25, lbd_penalty=1.0, rho=0.1,
                 K_internal=2, checkpoint=False, lambda_entropy=0.5, temperature=1.0, clip_grad=1.0,
                 use_bellman: bool = True,
                 use_LP: bool = True,
                 use_cosine: bool = True,
                 use_negative_cycle: bool = True,
                 use_delta_bf: bool = True,
                 use_flow: bool = True,
                 w_bellman: float = 1,
                 w_cosine: float = 1,
                 w_negcyc: float = 1,
                 w_negpen: float = 1,
                 w_deltabf: float = 1,
                 w_flow: float = 1,
                 ):
        super().__init__()
        self.lbd_penalty = lbd_penalty
        self.lambda_entropy = lambda_entropy
        #
        self.node_proj = Linear(in_dim, hidden2)
        self.bn_first = BN(hidden2)
        self.rank_mask = None  # used for topological ordering, None as initial value
        self.clip_grad = clip_grad
        # new
        self.w_bellman = w_bellman
        self.w_cosine = w_cosine
        self.w_negcyc = w_negcyc
        self.w_negpen = w_negpen
        self.w_deltabf = w_deltabf
        self.w_flow = w_flow
        # new
        self.convs = torch.nn.ModuleList([
            MultiHopEdgeGATConv_ordering(hidden2, edge_in_dim, K=K_internal,
                                         edge_hidden_dim=edge_hidden_dim,
                                         pos_dim=pos_dim,
                                         dropout=dropout,
                                         valid_hops=(1, 2, 3),
                                         checkpoint=checkpoint, topological_mask=self.rank_mask)
            for _ in range(num_layers)
        ])

        self.bns = torch.nn.ModuleList([BN(hidden2) for _ in range(num_layers)])

        self.test1 = Linear(hidden2, hidden2)
        self.test2 = Linear(hidden2, hidden2)

        self.lin_prob = Linear(hidden2, hidden1)
        self.out_prob = Linear(hidden1, 1)
        self.lin_val = Linear(hidden2, hidden1)
        self.out_val = Linear(hidden1, 1)
        # new
        self.node_score_mlp = Sequential(
            Linear(hidden2, hidden1),
            ReLU(),
            Linear(hidden1, 1)
        )

        self.temperature = temperature
        torch.nn.init.zeros_(self.out_prob.bias)

        self.edge_mlp = Sequential(
            Linear(2 * hidden2 + edge_in_dim + 2, edge_hidden_dim),
            ReLU(),
            Linear(edge_hidden_dim, 1)
        )

        self.edge_mlp_b = Sequential(
            Linear(2 * hidden2 + edge_in_dim + 1, edge_hidden_dim),
            ReLU(),
            Linear(edge_hidden_dim, 1)
        )

        # boolean
        self.use_bellman = use_bellman
        self.use_LP = use_LP
        self.use_cosine = use_cosine
        self.use_negative_cycle = use_negative_cycle
        self.use_delta_bf = use_delta_bf
        self.use_flow = use_flow

    def forward(self, data):
        node_attr, pos, edge_index, edge_attr, batch, spec = data.node_attr, data.pos, data.edge_index, data.edge_attr, data.batch, data.spec
        num_nodes, num_graphs = node_attr.size(0), batch.max().item() + 1

        x = F.relu(self.bn_first(self.node_proj(node_attr)))

        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x, edge_attr, node_scores, node_values = conv(x, pos, edge_index, edge_attr, topological_mask=None,
                                                          spec=spec)
            x = bn(x)
            x = F.relu(x + x_res)

        b = torch.zeros(num_nodes, device=x.device)
        for g in range(num_graphs):
            nodes = (batch == g).nonzero(as_tuple=True)[0]
            b[nodes[data.source[g]]] = 1.0
            b[nodes[data.sink[g]]] = -1.0

        # w_orig = data.raw_weight
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]

        from_val = node_values[edge_index[0]]
        to_val = node_values[edge_index[1]]

        from_val_ = from_val.unsqueeze(-1)  # [num_edges, 1]
        to_val_ = to_val.unsqueeze(-1)  # [num_edges, 1]

        edge_repr = torch.cat([x_src, edge_attr, x_dst, to_val_ - from_val_, to_val_], dim=-1)
        edge_logits = self.edge_mlp(edge_repr).squeeze(-1)

        edge_score = edge_logits
        edge_probs = torch.sigmoid(edge_score)

        return {
            "edge_probs": edge_probs,
            "node_values": node_values,
        }

def build_model(**common_kwargs):
    return ShortestPathGNN_edge_hop(
        **common_kwargs
    )
