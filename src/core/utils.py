import yaml
from box import ConfigBox


def load_params(params_file: str) -> ConfigBox:
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return ConfigBox(params)
