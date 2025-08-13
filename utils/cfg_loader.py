import yaml

def load_yaml(config_path:str):
    with open(config_path, 'r') as f:
        content = yaml.safe_load(f)

    return content