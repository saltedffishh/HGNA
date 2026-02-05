def knn_contrastive_loss(Z, knn_idx, temp=0.2):
    # Z: (n, d) normalized
    Z = torch.nn.functional.normalize(Z, dim=1)
    n, k = knn_idx.shape
    z_i = Z.unsqueeze(1).expand(-1, k, -1)     # (n,k,d)
    z_pos = Z[knn_idx]                         # (n,k,d)

    pos_sim = (z_i * z_pos).sum(-1) / temp     # (n,k)
    all_sim = (Z @ Z.T) / temp                 # (n,n)

    # log-softmax over all negatives
    log_prob = pos_sim - torch.logsumexp(all_sim, dim=1).unsqueeze(1)
    return -log_prob.mean()

def knn_mse_loss(Z, knn_idx):
    # Z: (n_nodes, d), knn_idx: (n_nodes, k)
    z_i = Z.unsqueeze(1).expand(-1, knn_idx.size(1), -1)  # (n,k,d)
    z_j = Z[knn_idx]                                      # (n,k,d)
    return ((z_i - z_j) ** 2).mean()
