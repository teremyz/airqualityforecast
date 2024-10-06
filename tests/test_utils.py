from box import ConfigBox

from src.core.utils import load_params


def test_load_params():
    params = load_params("config.yaml")

    assert isinstance(params, ConfigBox)
    assert "basic" in params.keys()
