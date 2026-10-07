import torch
import torch.nn as nn


class DirectedGCN(nn.Module):
    """Directed graph convolution with separate in/out neighbor aggregation.

    By default (``num_layers=1``) this is the original single-layer encoder used
    for all maritime experiments; the parameter names (``fc_in``, ``fc_out``,
    ``bias``, ``act``) are preserved so existing checkpoints load unchanged.

    Setting ``num_layers > 1`` stacks additional directed convolution layers
    (each ``out_ft -> out_ft``), reusing the same in/out adjacency matrices at
    every layer. A deeper stack aggregates over progressively larger
    neighborhoods, so on a very dense graph the node representations converge
    toward a shared summary.
    """

    def __init__(self, in_ft, out_ft, act, bias=True, num_layers=1):
        super(DirectedGCN, self).__init__()
        self.num_layers = num_layers

        # First layer (in_ft -> out_ft). Names preserved for backward compatibility.
        self.fc_in = nn.Linear(in_ft, out_ft, bias=False)  # For incoming edges
        self.fc_out = nn.Linear(in_ft, out_ft, bias=False)  # For outgoing edges
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        # Additional stacked layers (out_ft -> out_ft), only created when num_layers > 1.
        if num_layers > 1:
            self.hidden_fc_in = nn.ModuleList(
                [nn.Linear(out_ft, out_ft, bias=False) for _ in range(num_layers - 1)]
            )
            self.hidden_fc_out = nn.ModuleList(
                [nn.Linear(out_ft, out_ft, bias=False) for _ in range(num_layers - 1)]
            )
            self.hidden_act = nn.ModuleList(
                [nn.PReLU() if act == 'prelu' else act for _ in range(num_layers - 1)]
            )
            if bias:
                self.hidden_bias = nn.ParameterList(
                    [nn.Parameter(torch.zeros(out_ft)) for _ in range(num_layers - 1)]
                )
            else:
                self.hidden_bias = [None] * (num_layers - 1)

        for m in self.modules():
            self.weights_init(m)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _propagate(self, seq, adj_in, adj_out, sparse, fc_in, fc_out, bias, act):
        """One directed convolution: aggregate in- and out-neighbors, add bias, activate."""
        seq_fts_in = fc_in(seq)
        seq_fts_out = fc_out(seq)

        if sparse:
            out_in = torch.unsqueeze(torch.spmm(adj_in, torch.squeeze(seq_fts_in, 0)), 0)
            out_out = torch.unsqueeze(torch.spmm(adj_out, torch.squeeze(seq_fts_out, 0)), 0)
        else:
            out_in = torch.bmm(adj_in, seq_fts_in)
            out_out = torch.bmm(adj_out, seq_fts_out)

        # Combine the outputs (sum for equal weighting)
        out = out_in + out_out

        if bias is not None:
            out = out + bias

        return act(out)

    def forward(self, seq, adj_in, adj_out, sparse=False):
        h = self._propagate(seq, adj_in, adj_out, sparse,
                            self.fc_in, self.fc_out, self.bias, self.act)

        for i in range(self.num_layers - 1):
            h = self._propagate(h, adj_in, adj_out, sparse,
                                self.hidden_fc_in[i], self.hidden_fc_out[i],
                                self.hidden_bias[i], self.hidden_act[i])

        return h
