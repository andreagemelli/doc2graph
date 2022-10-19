from argparse import ArgumentParser
import os
from attrdict import AttrDict
import yaml

from src.paths import *

def create_folder(folder_path : str) -> None:
    """create a folder if not exists

    Args:
        folder_path (str): path
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return

def get_config(name : str) -> AttrDict:
    """get yaml config file

    Args:
        name (str): yaml file name without extension

    Returns:
        AttrDict: config
    """
    with open(CONFIGS / f'{name}.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
    return config

def project_tree() -> None:
    """ Create the project tree folder
    """
    create_folder(DATA)
    create_folder(OUTPUTS)
    create_folder(RUNS)
    create_folder(RESULTS)
    create_folder(TRAIN_SAMPLES)
    create_folder(TEST_SAMPLES)
    create_folder(CHECKPOINTS)
    return

def set_preprocessing(args: ArgumentParser) -> None:
    """ Set preprocessings args

    Args:
        args (ArgumentParser): 
    """
    with open(CONFIGS / 'base.yaml') as fileobj:
        cfg_preprocessing = dict(yaml.safe_load(fileobj))
    cfg_preprocessing['FEATURES']['add_geom'] = args.add_geom
    cfg_preprocessing['FEATURES']['add_embs'] = args.add_embs
    cfg_preprocessing['FEATURES']['add_hist'] = args.add_hist
    cfg_preprocessing['FEATURES']['add_visual'] = args.add_visual
    cfg_preprocessing['FEATURES']['add_eweights'] = args.add_eweights
    cfg_preprocessing['FEATURES']['num_polar_bins'] = args.num_polar_bins
    cfg_preprocessing['LOADER']['src_data'] = args.src_data
    cfg_preprocessing['GRAPHS']['data_type'] = args.data_type
    cfg_preprocessing['GRAPHS']['edge_type'] = args.edge_type
    cfg_preprocessing['GRAPHS']['node_granularity'] = args.node_granularity

    with open(CONFIGS / 'preprocessing.yaml', 'w') as f:
        yaml.dump(cfg_preprocessing, f)
    return
    