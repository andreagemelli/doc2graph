from pathlib import Path
from dotenv import dotenv_values
import os

# ROOT
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
config = dotenv_values(HERE / "root.env")
ROOT = Path(config['ROOT'])

# PROJECT TREE
DATA = ROOT / 'DATA'
CONFIGS = ROOT / 'configs'
OUTPUTS = ROOT / 'outputs'
WEIGHTS = OUTPUTS / 'weights'
RUNS = OUTPUTS / 'runs'
RESULTS = OUTPUTS / 'results'

# FUNSD
FUNSD_TRAIN = DATA / 'FUNSD' / 'training_data'
FUNSD_TEST = DATA / 'FUNSD' / 'testing_data'