import scipy.sparse as sp

def save_sparse_matrix(path, M):
    sp.save_npz(path, M)

def load_sparse_matrix(path):
    return sp.load_npz(path)
    
def maybe_load_or_build_knn(expr, stage_idx, exp_dir):
    graph_path = os.path.join(exp_dir, "graphs", f"knn_L_stage{stage_idx}.npz")

    if os.path.exists(graph_path):
        print("♻️  使用缓存 KNN 图")
        return load_sparse_matrix(graph_path)

    print("🔨 构建新的 KNN 图")
    L_G = build_knn_graph_from_expr(expr)
    save_sparse_matrix(graph_path, L_G)
    return L_G
