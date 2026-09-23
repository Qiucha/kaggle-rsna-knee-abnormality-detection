from functools import reduce
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

class Config:
    def __init__(self, path: str = ROOT / "config.yaml"):
        with open(path) as f:
            self.config = yaml.safe_load(f)

    def find(self, key: str, default=None):
        """
        Example for nested elements:
        
        get(key="data.dir")

        This returns config[data][dir]
        """
        return reduce(
            lambda d, k: d.get(k, default) if isinstance(d, dict) else default,
            key.split("."),
            self.config
        )
