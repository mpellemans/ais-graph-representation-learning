import torch
import torch.nn as nn


class EdgeConditionedDGCN(nn.Module):
    """Edge-conditioned Directed Graph Convolutional Network.

    Extends DirectedGCN with edge feature integration via a shared edge MLP
    that computes attention-like weights for neighbor aggregation.

    Message passing:
        H[i] = PReLU(in_agg(i) + out_agg(i) + W_self @ x_i + b)

    Where in_agg and out_agg use scatter_add_ with edge-MLP-derived weights,
    normalized per direction.
    """

    def __init__(self, in_ft, out_ft, act, edge_ft, bias=True):
        """
        Args:
            in_ft: Number of input features per node
            out_ft: Number of output features per node
            act: Activation function ('prelu' or nn.Module)
            edge_ft: Number of edge features
            bias: Whether to use bias
        """
        super(EdgeConditionedDGCN, self).__init__()

        self.fc_in = nn.Linear(in_ft, out_ft, bias=False)
        self.fc_out = nn.Linear(in_ft, out_ft, bias=False)
        self.fc_self = nn.Linear(in_ft, out_ft, bias=False)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_ft, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.bn = nn.BatchNorm1d(out_ft)

        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, edge_index, edge_features):
        """
        Args:
            seq: Node features, shape (1, N, in_ft)
            edge_index: Edge indices, shape (2, E) where [0]=source, [1]=target
            edge_features: Edge feature matrix, shape (E, edge_ft)

        Returns:
            Node embeddings, shape (1, N, out_ft)
        """
        # Remove the batch dimension for computation
        x = seq.squeeze(0)  # (N, in_ft)
        N = x.size(0)

        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)

        # Compute edge weights via shared MLP
        raw_w = self.edge_mlp(edge_features).squeeze(-1)  # (E,)

        # --- In-aggregation: for node i, aggregate from in-neighbors j (edges j->i) ---
        # dst is the target node; we aggregate messages from src nodes
        h_in = self.fc_in(x)  # (N, out_ft)
        msgs_in = raw_w.unsqueeze(-1) * h_in[src]  # (E, out_ft)

        # Normalize weights per target node
        w_sum_in = torch.zeros(N, device=x.device).scatter_add(0, dst, raw_w)
        w_norm_in = w_sum_in[dst].clamp(min=1e-8)  # (E,)
        msgs_in = msgs_in / w_norm_in.unsqueeze(-1)

        agg_in = torch.zeros(N, h_in.size(1), device=x.device)
        agg_in.scatter_add_(0, dst.unsqueeze(-1).expand_as(msgs_in), msgs_in)

        # --- Out-aggregation: for node i, aggregate from out-neighbors k (edges i->k) ---
        # src is the source node; we aggregate messages from dst nodes
        h_out = self.fc_out(x)  # (N, out_ft)
        msgs_out = raw_w.unsqueeze(-1) * h_out[dst]  # (E, out_ft)

        # Normalize weights per source node
        w_sum_out = torch.zeros(N, device=x.device).scatter_add(0, src, raw_w)
        w_norm_out = w_sum_out[src].clamp(min=1e-8)  # (E,)
        msgs_out = msgs_out / w_norm_out.unsqueeze(-1)

        agg_out = torch.zeros(N, h_out.size(1), device=x.device)
        agg_out.scatter_add_(0, src.unsqueeze(-1).expand_as(msgs_out), msgs_out)

        # --- Self-connection ---
        h_self = self.fc_self(x)  # (N, out_ft)

        # --- Combine ---
        out = agg_in + agg_out + h_self

        if self.bias is not None:
            out += self.bias

        out = self.bn(out)  # (N, out_ft)

        # Restore batch dimension
        return self.act(out.unsqueeze(0))
