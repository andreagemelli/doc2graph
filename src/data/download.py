import os
import requests
import zipfile

from src.paths import DATA

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def funsd():
    print("Downloading FUNSD")
    dlz = DATA / 'funsd.zip'
    download_url("https://guillaumejaume.github.io/FUNSD/dataset.zip", dlz)
    with zipfile.ZipFile(dlz, 'r') as zip_ref:
        zip_ref.extractall(DATA)
    os.remove(dlz)
    os.rename(DATA / 'dataset', DATA / 'FUNSD')
    return

def pau():
    #TODO dev data loader for Pau Riba's dataset
    print('Download  file: PAU Riba\'s dataset function download under dev.')

def get_data():
    funsd()
    pau()
    return