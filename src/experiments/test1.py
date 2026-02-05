# src/experiments/test.py
import argparse
import anndata
import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
from sklearn.decomposition import FastICA
import scanpy as sc

from model.hypergraph_nn import HypergraphEncoder, to_sparse_tensor
from model.cluster import hard_cluster_nodes, soft_cluster_hyperedges
from utils.paths import get_experiments_root, get_expr_cache_path, get_knn_cache_path
from utils.io import (
    load_sparse_matrix,
    load_expr_matrix,
    save_sparse_matrix,
    save_expr_matrix,
)
from utils.knn_build import build_knn_graph_from_expr
from datasets_loader_bar import load_dataset_pairs


def knn_mse_loss(Z, knn_idx):
    # Z: (n_nodes, d), knn_idx: (n_nodes, k)
    z_i = Z.unsqueeze(1).expand(-1, knn_idx.size(1), -1)
    z_j = Z[knn_idx]
    return ((z_i - z_j) ** 2).mean()

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


def laplacian_to_knn_idx(L, k=None):
    """
    从 KNN Laplacian 还原邻接并构造 knn_idx.
    L: scipy.sparse, 形状 (n,n)
    k: 若为 None，则用每行非零最小值作为 k
    """
    if not sp.isspmatrix(L):
        raise ValueError("L 必须是 scipy.sparse")

    L = L.tocsr()
    degrees = L.diagonal()
    A = sp.diags(degrees) - L
    A.eliminate_zeros()

    row_nnz = np.diff(A.indptr)
    valid = row_nnz[row_nnz > 0]
    if valid.size == 0:
        raise ValueError("KNN 邻接为空，无法构建 knn_idx")

    if k is None:
        k = int(valid.min())

    knn_idx = np.zeros((A.shape[0], k), dtype=np.int64)
    for i in range(A.shape[0]):
        start, end = A.indptr[i], A.indptr[i + 1]
        neighbors = A.indices[start:end]
        if neighbors.size >= k:
            knn_idx[i] = neighbors[:k]
        else:
            # 不足 k 时用自身补齐，避免 shape 崩溃
            pad = np.full(k - neighbors.size, i, dtype=np.int64)
            knn_idx[i] = np.concatenate([neighbors, pad], axis=0)

    return knn_idx


def pick_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # sparse mm 在 mps 上支持不完整，默认回退 cpu
        return torch.device("cpu")
    return torch.device("cpu")


def ensure_expr_cache(exp_dir, stage, dataset=None):
    expr_path = get_expr_cache_path(exp_dir, stage)
    if expr_path.exists():
        X, cells, genes = load_expr_matrix(expr_path)
        return X, cells, genes

    if dataset is None:
        raise FileNotFoundError(
            f"表达矩阵缓存不存在: {expr_path}. "
            "请提供 --dataset 以从原始数据构建。"
        )

    expr_path.parent.mkdir(parents=True, exist_ok=True)
    exprs, metas, names = load_dataset_pairs(dataset)
    if stage < 0 or stage >= len(exprs):
        raise IndexError(f"stage 超出范围: {stage}, 可用范围: [0, {len(exprs)-1}]")

    expr_df = exprs[stage]
    X = expr_df.T.values  # cells × genes
    cells = expr_df.columns.tolist()
    genes = expr_df.index.tolist()
    save_expr_matrix(expr_path, X, cells, genes)
    return X, cells, genes


def ensure_knn_laplacian(exp_dir, stage, X, cells, genes, k=15, pca_dim=50):
    knn_path = get_knn_cache_path(exp_dir, stage)
    if knn_path.exists():
        return load_sparse_matrix(knn_path)

    knn_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 utils.knn_build 中的构建逻辑
    expr_df = pd.DataFrame(X.T, index=genes, columns=cells)
    L_G = build_knn_graph_from_expr(expr_df, k=k, n_pca=pca_dim)
    save_sparse_matrix(knn_path, L_G)
    return L_G


def ensure_hypergraph(exp_dir, stage, X, n_programs=30, ica_max_iter=1000, abs_weight=False, keep_percentile=75.0):
    hyper_dir = exp_dir / "hypergraphs"
    hyper_dir.mkdir(exist_ok=True)

    H_path = hyper_dir / f"H_stage{stage}.npz"
    dv_path = hyper_dir / f"dv_stage{stage}.npy"
    de_path = hyper_dir / f"de_stage{stage}.npy"

    if H_path.exists() and dv_path.exists() and de_path.exists():
        H = load_sparse_matrix(H_path)
        dv = np.load(dv_path)
        de = np.load(de_path)
        return H, dv, de

    # 复用 utils/hypergraph_build.py 的同款逻辑
    ica = FastICA(
        n_components=n_programs,
        max_iter=ica_max_iter,
        random_state=0
    )
    Z = ica.fit_transform(X)  # cells × programs
    if abs_weight:
        Z = np.abs(Z)

    thresh = np.percentile(Z, keep_percentile)
    Z[Z < thresh] = 0.0

    H = sp.csr_matrix(Z)
    dv = np.array(H.sum(axis=1)).flatten()
    dv[dv == 0] = 1.0
    de = np.array(H.sum(axis=0)).flatten()
    de[de == 0] = 1.0

    save_sparse_matrix(H_path, H)
    np.save(dv_path, dv)
    np.save(de_path, de)

    return H, dv, de


def train_one_stage(
    exp_name,
    stage,
    hidden_dim=128,
    out_dim=64,
    k=15,
    pca_dim=50,
    n_programs=30,
    ica_max_iter=1000,
    abs_weight=False,
    keep_percentile=75.0,
    device_pref="auto",
    dataset=None,
    cluster_mode="hard_nodes",
    n_clusters=8,
    gmm_covariance="diag",
    gmm_prob_thresh=0.5,
    gmm_max_edges=None,
):
    exp_dir = get_experiments_root() / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {exp_dir}")

    # 1) 读取或构建表达矩阵 X（cells × genes），作为节点特征
    X, cells, genes = ensure_expr_cache(exp_dir, stage, dataset=dataset)

    # 2) 读取或构建超图 H/dv/de（ICA-based）
    H, dv, de = ensure_hypergraph(
        exp_dir,
        stage,
        X,
        n_programs=n_programs,
        ica_max_iter=ica_max_iter,
        abs_weight=abs_weight,
        keep_percentile=keep_percentile,
    )

    # 3) 读取或构建 KNN Laplacian，再转换为 knn_idx
    L_G = ensure_knn_laplacian(
        exp_dir,
        stage,
        X,
        cells,
        genes,
        k=k,
        pca_dim=pca_dim,
    )
    knn_idx = laplacian_to_knn_idx(L_G, k=k)

    device = pick_device(device_pref)

    model = HypergraphEncoder(X.shape[1], hidden_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("L_G nnz:", L_G.nnz)
    print("H nnz:", H.nnz)
    print("knn_idx unique per row:", np.mean([len(set(r)) for r in knn_idx]))
    

    H_sp = H  # keep scipy sparse for clustering
    H = to_sparse_tensor(H).coalesce().to(device)
    dv = torch.tensor(dv, dtype=torch.float32, device=device)
    de = torch.tensor(de, dtype=torch.float32, device=device)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    knn_idx = torch.tensor(knn_idx, dtype=torch.long, device=device)

    print("device:", device)
    print("X (cells × genes):", X.shape)
    print("H (cells × programs):", H.shape)
    print("knn_idx (cells × k):", knn_idx.shape)




    for epoch in range(50):
        Z = model(X, H, dv, de)
        loss = knn_contrastive_loss(Z, knn_idx,temp=0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # if epoch % 10 == 0:
        print(f"epoch {epoch} loss={loss.item():.10f}")
    
    print("Z mean:", Z.abs().mean().item())

    # -------------------------------------------------
    # 节点聚类（两种模式）
    # -------------------------------------------------
    Z_np = Z.detach().cpu().numpy()
    if cluster_mode == "hard_nodes":
        print("\n[Clustering] Hard clustering on nodes (KMeans)")
        node_labels = hard_cluster_nodes(Z_np, n_clusters=n_clusters, random_state=0)
        print("Node cluster sizes:", np.bincount(node_labels))
    elif cluster_mode == "soft_hyperedges":
        print("\n[Clustering] Soft clustering on hyperedges (GMM)")
        node_clusters, edge_probs = soft_cluster_hyperedges(
            Z_np,
            H_sp,
            de,
            n_clusters=n_clusters,
            covariance_type=gmm_covariance,
            prob_thresh=gmm_prob_thresh,
            max_edges=gmm_max_edges,
            random_state=0,
        )

        for c in range(n_clusters):
            print(f"Cluster {c}: nodes={len(node_clusters[c])}, edges~={np.sum(edge_probs[:, c] >= gmm_prob_thresh)}")
    else:
        raise ValueError(f"未知聚类模式: {cluster_mode}")


    #绘图
    # 1. 把学到的 Z 转成 AnnData
    adata = anndata.AnnData(X=Z.detach().cpu().numpy())

    # 2. 算一下邻居和 UMAP
    print("Running UMAP for visualization...")
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata)

    # 3. 画图 (如果有 label 信息最好，没有就看有没有聚类结构)
    # 如果你有原本的 cell_type 标签，可以加到 adata.obs['cell_type'] 里
    sc.pl.umap(adata, color=None, title="Learned Embeddings", save="_epoch50.png")
    print("UMAP saved to figures folder.")


def main():
    parser = argparse.ArgumentParser(description="Hypergraph + KNN 自监督最小测试")
    parser.add_argument("--exp_name", type=str, required=True, help="experiments/ 下实验目录名")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=15, help="KNN 的 k")
    parser.add_argument("--pca_dim", type=int, default=50, help="KNN PCA 维度")
    parser.add_argument("--n_programs", type=int, default=30, help="ICA 表达程序数量")
    parser.add_argument("--ica_max_iter", type=int, default=1000)
    parser.add_argument("--abs_weight", action="store_true", help="ICA 权重取绝对值")
    parser.add_argument("--keep_percentile", type=float, default=75.0, help="稀疏化百分位")
    parser.add_argument("--dataset", type=str, default=None, help="缺缓存时用于构建的原始数据集名")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--cluster_mode", type=str, default="hard_nodes", choices=["hard_nodes", "soft_hyperedges"])
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--gmm_covariance", type=str, default="diag", choices=["diag", "full", "tied", "spherical"])
    parser.add_argument("--gmm_prob_thresh", type=float, default=0.5)
    parser.add_argument("--gmm_max_edges", type=int, default=None)
    args = parser.parse_args()

    train_one_stage(
        exp_name=args.exp_name,
        stage=args.stage,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        k=args.k,
        pca_dim=args.pca_dim,
        n_programs=args.n_programs,
        ica_max_iter=args.ica_max_iter,
        abs_weight=args.abs_weight,
        keep_percentile=args.keep_percentile,
        device_pref=args.device,
        dataset=args.dataset,
        cluster_mode=args.cluster_mode,
        n_clusters=args.n_clusters,
        gmm_covariance=args.gmm_covariance,
        gmm_prob_thresh=args.gmm_prob_thresh,
        gmm_max_edges=args.gmm_max_edges,
    )


if __name__ == "__main__":
    main()
