import yaml
from pathlib import Path


def load_api_config():
    config_path = (
        Path(__file__).resolve().parent.parent / 'config' / 'api.yaml'
    )
    print(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
