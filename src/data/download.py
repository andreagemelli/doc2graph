import os
from src.paths import DATA
import requests
import zipfile
from git import Repo

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

def naf():
    print("Downloading NAF")
    Repo.clone_from("https://github.com/herobd/NAF_dataset.git", DATA / 'NAF')
    dlz = DATA / 'NAF' / "labeled_images.tar.gz"
    download_url("https://github.com/herobd/NAF_dataset/releases/download/v1.0/labeled_images.tar.gz", dlz)
    os.system(f'cd {DATA}/NAF && ./move_images.sh')

def get_data():
    #funsd()
    naf()
    return