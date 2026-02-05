# src/model/hypergraph_nn.py
import torch
import torch.nn as nn

def to_sparse_tensor(spm):
    spm = spm.tocoo()
    indices = torch.tensor([spm.row, spm.col], dtype=torch.long)
    values = torch.tensor(spm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, spm.shape)

class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, H, dv, de):
        # X: (n_nodes, in_dim)
        # H: sparse (n_nodes, n_edges)
        # dv, de: 1D tensors
        dv_inv_sqrt = torch.pow(dv, -0.5)
        de_inv = torch.pow(de, -1.0)

        # Dv^-1/2 X
        X = dv_inv_sqrt.unsqueeze(1) * X

        # H^T (Dv^-1/2 X)
        X = torch.sparse.mm(H.transpose(0,1), X)  # (n_edges, in_dim)
        X = de_inv.unsqueeze(1) * X               # De^-1
        X = torch.sparse.mm(H, X)                 # (n_nodes, in_dim)
        X = dv_inv_sqrt.unsqueeze(1) * X          # Dv^-1/2

        return self.lin(X)

class HypergraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.conv2 = HypergraphConv(hidden_dim, out_dim)

    def forward(self, X, H, dv, de):
        X = self.act(self.conv1(X, H, dv, de))
        X = self.conv2(X, H, dv, de)
        return X
