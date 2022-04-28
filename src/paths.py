from pathlib import Path

# Data folder
ROOT = Path(__file__).parent.parent
DATA = ROOT / 'dataset'
TRAIN = DATA / 'training_data'
TEST = DATA / 'testing_data'
CONFIGS = ROOT / 'configs'

