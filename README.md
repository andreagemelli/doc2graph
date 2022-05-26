# doc2graph - Documents transformed to Graphs

- [doc2graph - Documents transformed to Graphs](#doc2graph---documents-transformed-to-graphs)
  - [Info](#info)
  - [Install](#install)
  - [Settings](#settings)
  - [Training](#training)

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
python -m spacy download en_core_web_lg
```

## Settings
The project can be customized either changing directly `configs/base.yaml` file or providing these flags when calling `src/main.py`.
 - --add_embs: True / False (to add textual features to graph nodes)
 - --add_eweights:True / False (to add layout features to graph edges)
 - --add_visual: True / False (to add visual features to graph nodes)
 - --edge_type: (string) fully, knn or visibility to change the kind of connectivity
 - --src_data: (string) FUNSD, NAF or CUSTOM
 - --src_type: (string) img, pdf [if src_data is CUSTOM]

Change directly `configs/train.yaml` for training settings or create your own model copying `configs/models/gcn.yaml`.

If you want to use FUDGE pretrained model to detect entities and get their visual features:
- refer to [their github](https://github.com/herobd/FUDGE) for installation and weights download
- replace `run.py` and `model/yolo_box_detector.py` scripts with our custom version (TODO: link download)

## Training
1. To download data and init project,
```
python src/main.py --init
```
2. To train a **GCN** model for **Entity Labeling** on FUNSD (using CPU):
```
python src/main.py
```
3. To test a trained **GCN** model for **Entity Labeling** on FUNSD (using GPU):
```
python src/main.py --gpu 0 --model gcn --test -w output/weights/gcn-1651069965.pt
```
