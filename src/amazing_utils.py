import os

from attrdict import AttrDict
import yaml

from src.paths import *

def create_folder(folder_path : str):
    """create a folder if not exists

    Args:
        folder_path (str): path
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

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

def project_tree():
    create_folder(DATA)
    create_folder(OUTPUTS)
    create_folder(WEIGHTS)
    create_folder(RUNS)
    create_folder(RESULTS)
    return

def set_preprocessing(add_embs, add_visual, add_eweights, src_data, data_type, edge_type):
    with open(CONFIGS / 'base.yaml') as fileobj:
        cfg_preprocessing = dict(yaml.safe_load(fileobj))
    cfg_preprocessing['FEATURES']['add_embs'] = add_embs
    cfg_preprocessing['FEATURES']['add_visual'] = add_visual
    cfg_preprocessing['FEATURES']['add_eweights'] = add_eweights
    cfg_preprocessing['LOADER']['src_data'] = src_data
    cfg_preprocessing['GRAPHS']['data_type'] = data_type
    cfg_preprocessing['GRAPHS']['edge_type'] = edge_type
    # print(cfg_preprocessing)

    with open(CONFIGS / 'preprocessing.yaml', 'w') as f:
        yaml.dump(cfg_preprocessing, f)
    return config
    