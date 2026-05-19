# scripts/run_evolution.py

# ===== Global argparse patch for Python 3.14 compatibility =====
import argparse
_original_add_argument = argparse.ArgumentParser.add_argument
def _patched_add_argument(self, *args, **kwargs):
    if 'help' in kwargs and not isinstance(kwargs['help'], str):
        kwargs['help'] = str(kwargs['help'])
    return _original_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _patched_add_argument
# ===== End patch =====

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig
from src.evolution.loop import AdversarialLoop

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    loop = AdversarialLoop(cfg)
    loop.run()

if __name__ == "__main__":
    main()