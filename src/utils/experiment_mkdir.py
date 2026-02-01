# src/utils/experiment_mkdir.py

import os
import datetime

def create_experiment_dir(base_dir, config):
    """
    base_dir: experiments/
    config: dict of experiment params
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg_str = "_".join([f"{k}{v}" for k, v in config.items()])
    exp_name = f"{timestamp}__{cfg_str}"

    exp_path = os.path.join(base_dir, exp_name)
    os.makedirs(exp_path, exist_ok=False)

    # 子目录
    os.makedirs(os.path.join(exp_path, "data"))
    os.makedirs(os.path.join(exp_path, "graphs"))
    os.makedirs(os.path.join(exp_path, "results"))

    return exp_path
