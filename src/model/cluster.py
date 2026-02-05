import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def hard_cluster_nodes(Z, n_clusters=8, random_state=0):
    """
    对节点嵌入做硬聚类（KMeans）。

    Parameters
    ----------
    Z : np.ndarray, shape (n_nodes, d)
    n_clusters : int
    random_state : int

    Returns
    -------
    labels : np.ndarray, shape (n_nodes,)
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Z)
    return labels


def soft_cluster_hyperedges(
    Z,
    H_sp,
    de,
    n_clusters=8,
    covariance_type="diag",
    prob_thresh=0.5,
    max_edges=None,
    random_state=0,
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

    Returns
    -------
    node_clusters : dict[int, np.ndarray]
        每个聚类对应的节点索引集合
    edge_resp : np.ndarray, shape (n_edges, n_clusters)
        超边对每个簇的软归属概率
    """
    # 超边虚拟顶点嵌入：H^T @ Z / de
    edge_emb = H_sp.T.dot(Z)
    de_safe = np.asarray(de).reshape(-1, 1)
    edge_emb = edge_emb / de_safe

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    gmm.fit(edge_emb)
    edge_probs = gmm.predict_proba(edge_emb)

    node_clusters = {}
    for c in range(n_clusters):
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

    return node_clusters, edge_probs
