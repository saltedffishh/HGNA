import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch




def hard_cluster_nodes(Z, n_clusters=8, random_state=0, resolution=None):
    """
    对节点嵌入做硬聚类（Leiden）。

    Parameters
    ----------
    Z : np.ndarray, shape (n_nodes, d)
    n_clusters : int
    random_state : int
    resolution : float or None

    Returns
    -------
    labels : np.ndarray, shape (n_nodes,)
    """
    # Leiden uses a resolution parameter; if not provided, map n_clusters to a reasonable default.
    if resolution is None:
        resolution = max(0.1, n_clusters / 10.0)

    adata = sc.AnnData(X=Z.astype(np.float32, copy=False))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    labels = adata.obs["leiden"].astype(int).to_numpy()
    return labels

    # KMeans (legacy hard clustering)
    # km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    # labels = km.fit_predict(Z)
    # return labels


def soft_cluster_hyperedges(
    Z,
    H_sp,
    de,
    n_clusters=8,
    covariance_type="diag",
    prob_thresh=0.5,
    max_edges=None,
    random_state=0,
    select_n_clusters=None,
    min_clusters=2,
    max_clusters=30,
):
    """
    先将超边内节点嵌入聚合成虚拟顶点，再对虚拟顶点做软聚类，
    最后在同一聚类内对超边相连节点取交集得到节点簇。

    Parameters
    ----------
    Z : np.ndarray, shape (n_nodes, d)
    H_sp : scipy.sparse, shape (n_nodes, n_edges)
    de : np.ndarray, shape (n_edges,)
    n_clusters : int
    covariance_type : str
    prob_thresh : float
    max_edges : int or None
    random_state : int
    select_n_clusters : {"bic", "aic"} or None
        若指定，则在 [min_clusters, max_clusters] 范围内自动选择簇数。
    min_clusters : int
    max_clusters : int

    Returns
    -------
    node_clusters : dict[int, np.ndarray]
        每个聚类对应的节点索引集合
    edge_resp : np.ndarray, shape (n_edges, n_clusters)
        超边对每个簇的软归属概率
    """
    # 超边虚拟顶点嵌入：H^T @ Z / de
    edge_emb = H_sp.T.dot(Z)
    #de_safe = np.asarray(de).reshape(-1, 1)
    if torch.is_tensor(de):
        de_cpu = de.detach().cpu().numpy()
    else:
        de_cpu = de
    de_safe = np.asarray(de_cpu).reshape(-1, 1)
    edge_emb = edge_emb / de_safe

    if select_n_clusters is None:
        select_n_clusters = "bic"

    best_score = float("inf")
    best_gmm = None
    for k in range(min_clusters, max_clusters + 1):
        gmm_k = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm_k.fit(edge_emb)
        score = gmm_k.bic(edge_emb) if select_n_clusters == "bic" else gmm_k.aic(edge_emb)
        if score < best_score:
            best_score = score
            best_gmm = gmm_k
    gmm = best_gmm
    edge_probs = gmm.predict_proba(edge_emb)
    n_clusters_used = gmm.n_components

    # Legacy fixed-n_clusters GMM (commented out)
    # gmm = GaussianMixture(
    #     n_components=n_clusters,
    #     covariance_type=covariance_type,
    #     random_state=random_state,
    # )
    # gmm.fit(edge_emb)
    # edge_probs = gmm.predict_proba(edge_emb)

    node_clusters = {}
    for c in range(n_clusters_used):
        edges = np.where(edge_probs[:, c] >= prob_thresh)[0]
        if max_edges is not None and edges.size > max_edges:
            top_idx = np.argsort(edge_probs[:, c])[-max_edges:]
            edges = top_idx

        if edges.size == 0:
            node_clusters[c] = np.array([], dtype=np.int64)
            continue

        H_sub = H_sp[:, edges]
        counts = np.array((H_sub > 0).sum(axis=1)).flatten()
        nodes = np.where(counts == edges.size)[0]
        node_clusters[c] = nodes

    return node_clusters, edge_probs, n_clusters_used
