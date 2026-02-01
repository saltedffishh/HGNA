# src/utils/paths.py

import os
from pathlib import Path
import datetime

# -------------------------------------------------
# 1. 项目根目录（唯一入口）
# -------------------------------------------------

def get_project_root() -> Path:
    """
    返回 HGNA/ 项目根目录
    假设本文件位于: HGNA/src/utils/paths.py
    """
    return Path(__file__).resolve().parents[2]


# -------------------------------------------------
# 2. 固定一级目录
# -------------------------------------------------

def get_src_dir() -> Path:
    return get_project_root() / "src"

def get_experiments_root() -> Path:
    path = get_project_root() / "experiments"
    path.mkdir(exist_ok=True)
    return path

def get_data_root() -> Path:
    return get_project_root() / "COVID19_data"


# -------------------------------------------------
# 3. 实验目录管理
# -------------------------------------------------

def create_experiment_dir(config: dict) -> Path:
    """
    在 experiments/ 下创建新的实验目录
    名字 = 时间 + 配置
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg_str = "_".join([f"{k}{v}" for k, v in config.items()])

    exp_name = f"{timestamp}__{cfg_str}"
    exp_dir = get_experiments_root() / exp_name

    # 创建子目录
    (exp_dir / "data").mkdir(parents=True)
    (exp_dir / "graphs").mkdir()
    (exp_dir / "results").mkdir()
    (exp_dir / "logs").mkdir()

    return exp_dir


# -------------------------------------------------
# 4. 实验内部标准路径（强烈推荐）
# -------------------------------------------------

def get_expr_cache_path(exp_dir: Path, stage_idx: int) -> Path:
    return exp_dir / "data" / f"expr_stage{stage_idx}.npz"

def get_knn_cache_path(exp_dir: Path, stage_idx: int) -> Path:
    return exp_dir / "graphs" / f"knn_L_stage{stage_idx}.npz"

def get_hypergraph_cache_path(exp_dir: Path, stage_idx: int) -> Path:
    return exp_dir / "graphs" / f"hypergraph_H_stage{stage_idx}.npz"

def get_config_path(exp_dir: Path) -> Path:
    return exp_dir / "config.yaml"

def get_log_path(exp_dir: Path) -> Path:
    return exp_dir / "logs" / "run.log"
