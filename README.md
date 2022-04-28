# doc2graph - Documents transformed to Graphs
- [doc2graph - Documents transformed to Graphs](#doc2graph---documents-transformed-to-graphs)
  - [Info](#info)
  - [Install](#install)
  - [Results](#results)
  - [Usage](#usage)

## Info
Library to convert documents to graphs and perform several tasks on different datasets, like KIE on FUNSD.

## Install
First, install [setuptools-git-versioning](https://pypi.org/project/setuptools-git-versioning/) and doc2graph package itself.
```
cd doc2graph
pip install setuptools-git-versioning && pip install -e .
```
Then, install [torch](https://pytorch.org/get-started/locally/), [dgl](https://www.dgl.ai/pages/start.html) and [spacy](https://spacy.io/usage/models#quickstart) dependencies (refer to their get started sections to install different version)
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
python -m spacy download en_core_web_sm
```
Edit the 6th line on `src/paths` giving your absolute folder 

## Results

## Usage
1. To train a **GCN** model for **Entity Labeling** on FUNSD (using CPU):
```
python src/main.py
```

2. To test a trained **GCN** model for **Entity Labeling** on FUNSD (using GPU):
```
python src/main.py --gpu 0 --config gcn --test -w output/weights/gcn-1651069965.pt
```