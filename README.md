# <p align=center>`Doc2Graph`</p> 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/doc2graph-a-task-agnostic-document/entity-linking-on-funsd)](https://paperswithcode.com/sota/entity-linking-on-funsd?p=doc2graph-a-task-agnostic-document) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/doc2graph-a-task-agnostic-document/semantic-entity-labeling-on-funsd)](https://paperswithcode.com/sota/semantic-entity-labeling-on-funsd?p=doc2graph-a-task-agnostic-document)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This library is the implementation of the paper [Doc2Graph: a Task Agnostic Document Understanding Framework based on Graph Neural Networks](https://arxiv.org/abs/2208.11168), accepted at [TiE @ ECCV 2022](https://sites.google.com/view/tie-eccv2022/accepted-papers?authuser=0).

The model and pipeline aims at being task-agnostic on the domain of Document Understanding. It is an ongoing project, these are the steps already achieved and the ones we would like to implement in the future:

- [x] Build a model based on GNNs to spot key-value relationships on forms
- [x] Publish the preliminary results and the code
- [x] Extend the framework to other document-related tasks
  - [x] Business documents Layout Analysis
  - [x] Table Detection
- [ ] Let the user train Doc2Graph over private / other datasets using our dataloader
- [ ] Transform Doc2Graph into a PyPI package

Roadmap:
- [`Doc2Graph`](#doc2graph)
  - [Environment Setup](#environment-setup)
  - [Training](#training)
  - [Testing](#testing)
  - [Cite this project](#cite-this-project)

## Environment Setup
Setup the initial conda environment

```
conda create -n doc2graph python=3.9 ipython cudatoolkit=11.3 -c anaconda &&
conda activate doc2graph &&
cd doc2graph
```

Then, install [setuptools-git-versioning](https://pypi.org/project/setuptools-git-versioning/) and doc2graph package itself. The following has been tested only on linux: for different OS installations refer directly to [PyTorch](https://pytorch.org/get-started/previous-versions/) and [DGL](https://www.dgl.ai/pages/start.html) original documentation.

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url &&
https://download.pytorch.org/whl/cu113 &&
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html &&
pip install setuptools-git-versioning && pip install -e . &&
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0.tar.gz
```

Finally, create the project folder structure and download data:

```
python src/init.py
```
The script will download and setup:
- FUNSD and the 'adjusted_annotations' for FUNSD[^1] are given by the work of[^3].
- The yolo detection bbox described in the paper (If you want to use YOLOv5-small to detect entities, script in `notebooks/YOLO.ipynb`, refer to [their github](https://github.com/ultralytics/yolov5) for the installation. Clone the repository into `src/models/yolov5`).
- The Pau Riba's[^2] dataset with our train / test split.

[^1]: G. Jaume et al., FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents, ICDARW 2019
[^2]: P. Riba et al, Table Detection in Invoice Documents by Graph Neural Networks, ICDAR 2019
[^3]: Hieu M. Vu et al., REVISING FUNSD DATASET FOR KEY-VALUE DETECTION IN DOCUMENT IMAGES, arXiv preprint 2020

**Checkpoints**
You can download our model checkpoints [here](https://drive.google.com/file/d/15jKWYLTcb8VwE7XY_jcRvZTAmqy_ElJ_/view?usp=sharing). Place them into `src/models/checkpoints`.

---
## Training
1. To train yours **Doc2Graph** model (using CPU) use:
```
python src/main.py [SETTINGS]
```
2. Instead, to test a trained **Doc2Graph** model (using GPU) [weights can be one or more file]:
```
python src/train.py [SETTINGS] --gpu 0 --test --weights *.pt
```
The project can be customized either changing directly `configs/base.yaml` file or providing these flags when calling `src/main.py`.

**Features**
 - --add-geom: True / False (to add positional features to graph nodes)
 - --add-embs: True / False (to add textual features to graph nodes)
 - --add-hist: True / False (to add visual features to graph nodes)
 - --add-visual: True / False (to add visual features to graph nodes)
 - --add-eweights: True / False (to add polar relative coordinates between nodes to graph edges)

**Data**
 - --src-data: string [FUNSD, PAU or CUSTOM] (CUSTOM still under dev)
 - --src-type: string [img, pdf] (if src_data is CUSTOM, still under dev)

**Graphs**
 - --edge-type: string [fully, knn] (to change the kind of connectivity)
 - --node-granularity: string [gt, yolo, ocr] (choose the granularity of nodes to be used, gt (if given), ocr (words) or yolo (entities))
 - --num-polar-bins: int [Default 8] (number of bins into which discretize the space for edge polar features. It must be a power of 2)

Change directly `configs/train.yaml` for training settings or pass these flags to `src/main.py`. To create your own model (changing hyperparams) copy `configs/models/*.yaml`. 

**Training/Testing**
- --gpu: int [Default -1] (which GPU to use. Set -1 to use CPU)
- --test: true / false (skip training if true)
- --weights: strin(s) (provide weight file(s) relative path(s), if testing)

## Testing

You can use our pretrained models over the test sets of FUNSD[^1] and Pau Riba's[^2] datasets.

1. On FUNSD we were able to perform both Semantic Entity Labeling and Entity Linking:

**E2E-FUNSD-GT**:
```
python src/main.py -addG -addT -addE -addV --gpu 0 --test --weights e2e-funsd-best.pt
```

**E2E-FUNSD-YOLO**:
```
python src/main.py -addG -addT -addE -addV --gpu 0 --test --weights e2e-funsd-best.pt --node-granularity yolo
```

2. on Pau Riba's dataset, we were able to perform both Layout Analysis and Table Detection

**E2E-PAU**:
```
python src/main.py -addG -addT -addE -addV --gpu 0 --test --weights e2e-pau-best.pt --src-data PAU --edge-type knn
```
  
---
## Cite this project
If you want to use our code in your project(s), please cite us:
```
@misc{https://doi.org/10.48550/arxiv.2208.11168,
  doi = {10.48550/ARXIV.2208.11168},
  url = {https://arxiv.org/abs/2208.11168},
  author = {Gemelli, Andrea and Biswas, Sanket and Civitelli, Enrico and Lladós, Josep and Marinai, Simone},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Doc2Graph: a Task Agnostic Document Understanding Framework based on Graph Neural Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
