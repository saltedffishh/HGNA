import yaml

def save_config(path, config):
    with open(path, "w") as f:
        yaml.dump(config, f)

