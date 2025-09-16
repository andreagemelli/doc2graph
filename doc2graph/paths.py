from pathlib import Path

# ROOT - dynamically determined from the location of this file
# This file is in doc2graph/, so ROOT is the parent directory of doc2graph/
HERE = Path(__file__).parent
ROOT = HERE.parent

# PROJECT TREE
DATA = ROOT / "DATA"
CONFIGS = ROOT / "configs"
CFGM = CONFIGS / "models"
OUTPUTS = ROOT / "outputs"
RUNS = OUTPUTS / "runs"
RESULTS = OUTPUTS / "results"
IMGS = OUTPUTS / "images"
TRAIN_SAMPLES = OUTPUTS / "train_samples"
TEST_SAMPLES = OUTPUTS / "test_samples"
TRAINING = ROOT / "doc2graph" / "training"
MODELS = ROOT / "doc2graph" / "models"
CHECKPOINTS = MODELS / "checkpoints"
INFERENCE = ROOT / "inference"

# FUNSD
FUNSD_TRAIN = DATA / "FUNSD" / "training_data"
FUNSD_TEST = DATA / "FUNSD" / "testing_data"

# PAU
PAU_TRAIN = DATA / "PAU" / "train"
PAU_TEST = DATA / "PAU" / "test"
