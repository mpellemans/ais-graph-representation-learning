import torch.nn as nn
from ..layers import GCN, DirectedGCN, EdgeConditionedDGCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, directed=False, edge_dim=None, num_layers=1):
        super(DGI, self).__init__()
        self.directed = directed
        self.use_edge_features = directed and edge_dim is not None

        if self.use_edge_features:
            self.encoder = EdgeConditionedDGCN(n_in, n_h, activation, edge_ft=edge_dim)
        elif directed:
            self.encoder = DirectedGCN(n_in, n_h, activation, num_layers=num_layers)
        else:
            self.encoder = GCN(n_in, n_h, activation)

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2,
                adj_out=None, edge_index=None, edge_features=None):
        if self.use_edge_features:
            h_1 = self.encoder(seq1, edge_index, edge_features)
            h_2 = self.encoder(seq2, edge_index, edge_features)
        elif self.directed:
            h_1 = self.encoder(seq1, adj, adj_out, sparse)
            h_2 = self.encoder(seq2, adj, adj_out, sparse)
        else:
            h_1 = self.encoder(seq1, adj, sparse)
            h_2 = self.encoder(seq2, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    def embed(self, seq, adj, sparse, msk, adj_out=None,
              edge_index=None, edge_features=None):
        if self.use_edge_features:
            h_1 = self.encoder(seq, edge_index, edge_features)
        elif self.directed:
            h_1 = self.encoder(seq, adj, adj_out, sparse)
        else:
            h_1 = self.encoder(seq, adj, sparse)

        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

