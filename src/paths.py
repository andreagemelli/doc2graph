from pathlib import Path
from dotenv import dotenv_values
import os

# Data folder

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
config = dotenv_values(HERE / "root.env")
# ROOT = '/home/gemelli/projects/doc2graph'
ROOT = Path(config['ROOT'])
DATA = ROOT / 'dataset'
CONFIGS = ROOT / 'configs'

# FUNSD
FUNSD_TRAIN = DATA / 'FUNSD' / 'training_data'
FUNSD_TEST = DATA / 'FUNSD' / 'testing_data'