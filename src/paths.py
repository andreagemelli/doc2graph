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
MODELS = CONFIGS / 'models'
OUTPUTS = ROOT / 'outputs'
WEIGHTS = OUTPUTS / 'weights'
RUNS = OUTPUTS / 'runs'
RESULTS = OUTPUTS / 'results'
IMGS = OUTPUTS / 'images'
TRAIN_SAMPLES = OUTPUTS / 'train_samples'
TEST_SAMPLES = OUTPUTS / 'test_samples'
TRAINING = ROOT / 'src' / 'training'
CHECKPOINTS = TRAINING / 'checkpoints'

# FUNSD
FUNSD_TRAIN = DATA / 'FUNSD' / 'training_data'
FUNSD_TEST = DATA / 'FUNSD' / 'testing_data'

# NAF
NAF = DATA / 'NAF'
SIMPLE_NAF_TRAIN = NAF / 'simple' / 'train'
SIMPLE_NAF_VALID = NAF / 'simple' / 'valid'
SIMPLE_NAF_TEST = NAF / 'simple' / 'test'

# FUDGE
FUDGE = ROOT / 'FUDGE'