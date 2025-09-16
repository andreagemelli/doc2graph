# <p align=center>`Doc2Graph`</p> 

![model](model.png)

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
- [ ] Publish Doc2Graph to PyPI for easy installation

## Quick Start

Get up and running with Doc2Graph in minutes:

```bash
# Clone and install
git clone https://github.com/andreagemelli/doc2graph.git
cd doc2graph
uv sync
uv pip install -e .

# Initialize the project (downloads datasets)
uv run doc2graph.main --init

# Run inference on a document
uv run python -m doc2graph.main -addG -addT -addE -addV --weights e2e-funsd-best.pt --inference --docs /path/to/your/image.png
```

Check out the [tutorial notebook](tutorial/kie.ipynb) for a complete walkthrough!

Index:
- [`Doc2Graph`](#doc2graph)
  - [News!](#news)
  - [Environment Setup](#environment-setup)
  - [Training](#training)
  - [Testing](#testing)
  - [Cite this project](#cite-this-project)

---
## News!
- 🔥 Added **inference** method: you can now use Doc2Graph directly on your documents simply passing a path to them! <br> This call will output an image with the connected entities and a json / dictionary with all the useful information you need! 🤗
```
uv run python -m doc2graph.main -addG -addT -addE -addV --weights e2e-funsd-best.pt --inference --docs /path/to/your/image.png
```
  
- 🔥 Added **tutorial** folder: get to know how to use Doc2Graph from the tutorial notebooks!

## Environment Setup

### Prerequisites
- Python 3.8-3.11 (recommended: 3.10)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv** (if not already installed):
```bash
pip install uv
```

2. **Clone and setup the project**:
```bash
git clone https://github.com/andreagemelli/doc2graph.git
cd doc2graph
```

3. **Install the package in development mode**:
```bash
uv sync
uv pip install -e .
```

This will:
- Install all dependencies with compatible versions
- Install the `doc2graph` package in development mode
- Set up the virtual environment with Python 3.10

### Additional Setup (Optional)

For GPU acceleration, you may need to install CUDA-specific versions of PyTorch and DGL:

```bash
# For CUDA 11.8 (adjust version as needed)
uv add torch==1.13.1+cu118 torchvision==0.14.1+cu118 --index-url https://download.pytorch.org/whl/cu118
uv add dgl-cu118 --index-url https://data.dgl.ai/wheels/repo.html
```

**Note**: For different OS installations or CUDA versions, refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) and [DGL](https://www.dgl.ai/pages/start.html) documentation.

Finally, create the project folder structure and download data:

```
python doc2graph/main.py --init
```
The script will download and setup:
- FUNSD and the 'adjusted_annotations' for FUNSD[^1] are given by the work of[^3].
- The yolo detection bbox described in the paper (If you want to use YOLOv5-small to detect entities, script in `notebooks/YOLO.ipynb`, refer to [their github](https://github.com/ultralytics/yolov5) for the installation. Clone the repository into `doc2graph/models/yolov5`).
- The Pau Riba's[^2] dataset with our train / test split.

[^1]: G. Jaume et al., FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents, ICDARW 2019
[^2]: P. Riba et al, Table Detection in Invoice Documents by Graph Neural Networks, ICDAR 2019
[^3]: Hieu M. Vu et al., REVISING FUNSD DATASET FOR KEY-VALUE DETECTION IN DOCUMENT IMAGES, arXiv preprint 2020

**Checkpoints**
You can download our model checkpoints [here](https://drive.google.com/file/d/15jKWYLTcb8VwE7XY_jcRvZTAmqy_ElJ_/view?usp=sharing). Place them into `doc2graph/models/checkpoints`.

---
## Training
1. To train our **Doc2Graph** model (using CPU) use:
```
python doc2graph/main.py [SETTINGS]
```
2. Instead, to test a trained **Doc2Graph** model (using GPU) [weights can be one or more file]:
```
python doc2graph/main.py [SETTINGS] --gpu 0 --test --weights *.pt
```
The project can be customized either changing directly `configs/base.yaml` file or providing these flags when calling `doc2graph/main.py`.

**Features**
 - --add-geom: bool (to add positional features to graph nodes)
 - --add-embs: bool (to add textual features to graph nodes)
 - --add-hist: bool (to add visual features to graph nodes)
 - --add-visual: bool (to add visual features to graph nodes)
 - --add-eweights: bool (to add polar relative coordinates between nodes to graph edges)

**Data**
 - --src-data: string [FUNSD, PAU or CUSTOM] (CUSTOM still under dev)
 - --src-type: string [img, pdf] (if src_data is CUSTOM, still under dev)

**Graphs**
 - --edge-type: string [fully, knn] (to change the kind of connectivity)
 - --node-granularity: string [gt, yolo, ocr] (choose the granularity of nodes to be used, gt (if given), ocr (words) or yolo (entities))
 - --num-polar-bins: int [Default 8] (number of bins into which discretize the space for edge polar features. It must be a power of 2)

 **Inference (only for KiE)**
 - --inference: bool (run inference on given document/s path/s)
 - --docs: list (list your absolute path to your document)

Change directly `configs/train.yaml` for training settings or pass these flags to `doc2graph/main.py`. To create your own model (changing hyperparams) copy `configs/models/*.yaml`. 

**Training/Testing**
- --model: string [e2e, edge, gcn] (which model to use, which yaml file to load)
- --gpu: int [Default -1] (which GPU to use. Set -1 to use CPU)
- --test: true / false (skip training if true)
- --weights: strin(s) (provide weight file(s) relative path(s), if testing)

## Testing

You can use our pretrained models over the test sets of FUNSD[^1] and Pau Riba's[^2] datasets.

1. On FUNSD we were able to perform both Semantic Entity Labeling and Entity Linking:

**E2E-FUNSD-GT**:
```
python doc2graph/main.py -addG -addT -addE -addV --gpu 0 --test --weights e2e-funsd-best.pt
```

**E2E-FUNSD-YOLO**:
```
python doc2graph/main.py -addG -addT -addE -addV --gpu 0 --test --weights e2e-funsd-best.pt --node-granularity yolo
```

2. on Pau Riba's dataset, we were able to perform both Layout Analysis and Table Detection

**E2E-PAU**:
```
python doc2graph/main.py -addG -addT -addE -addV --gpu 0 --test --weights e2e-pau-best.pt --src-data PAU --edge-type knn
```
  
---
## Cite this project
If you want to use our code in your project(s), please cite us:
```
@InProceedings{10.1007/978-3-031-25069-9_22,
author="Gemelli, Andrea
and Biswas, Sanket
and Civitelli, Enrico
and Llad{\'o}s, Josep
and Marinai, Simone",
editor="Karlinsky, Leonid
and Michaeli, Tomer
and Nishino, Ko",
title="Doc2Graph: A Task Agnostic Document Understanding Framework Based on Graph Neural Networks",
booktitle="Computer Vision -- ECCV 2022 Workshops",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="329--344",
abstract="Geometric Deep Learning has recently attracted significant interest in a wide range of machine learning fields, including document analysis. The application of Graph Neural Networks (GNNs) has become crucial in various document-related tasks since they can unravel important structural patterns, fundamental in key information extraction processes. Previous works in the literature propose task-driven models and do not take into account the full power of graphs. We propose Doc2Graph, a task-agnostic document understanding framework based on a GNN model, to solve different tasks given different types of documents. We evaluated our approach on two challenging datasets for key information extraction in form understanding, invoice layout analysis and table detection. Our code is freely accessible on https://github.com/andreagemelli/doc2graph.",
isbn="978-3-031-25069-9"
}
```
