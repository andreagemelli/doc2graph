# doc2graph - Documents transformed to Graphs

- [doc2graph - Documents transformed to Graphs](#doc2graph---documents-transformed-to-graphs)
  - [Info](#info)
  - [Install](#install)
  - [Training](#training)
    - [Settings](#settings)
  - [Testing](#testing)
    - [FUNSD](#funsd)
    - [NAF](#naf)

## Info
Library to convert documents to graphs and perform several tasks on different datasets, e.g. Key Information Extraction on FUNSD.

## Install
First, install [setuptools-git-versioning](https://pypi.org/project/setuptools-git-versioning/) and doc2graph package itself.

```
create -n doc2graph python=3.9 ipython cudatoolkit=11.3 -c anaconda
conda activate doc2graph
cd doc2graph
pip install setuptools-git-versioning && pip install -e .
python -m spacy download en_core_web_lg
```

---
## Training
1. To download data and init project,
```
python src/main.py --init
```
2. To train a **GCN** model for **Entity Labeling** on FUNSD (using CPU):
```
python src/main.py [SETTINGS]
```
3. To test a trained **GCN** model for **Entity Labeling** on FUNSD (using GPU):
```
python src/main.py [SETTINGS] --gpu 0 --test --weights node.pt
```

### Settings
The project can be customized either changing directly `configs/base.yaml` file or providing these flags when calling `src/main.py`.

**Features**
 - --add_embs: True / False (to add textual features to graph nodes)
 - --add_eweights: True / False (to add polar relative coordinates between nodes to graph edges)
 - --add_visual: True / False (to add visual features to graph nodes)
 - --add_fudge: True / False (to add fudge features to graph nodes)
 - --add_histogram: True / False (to add visual features to graph nodes)
 - --add_geom: True / False (to add positional features to graph nodes)

**Others**
 - --edge_type: (string) fully or knn to change the kind of connectivity
 - --src_data: (string) FUNSD or NAF [or CUSTOM]
 - --src_type: (string) img, pdf [if src_data is CUSTOM]

Change directly `configs/train.yaml` for training settings or create your own model copying `configs/models/gcn.yaml`.

If you want to use FUDGE pretrained model to detect entities and get their visual features:
- refer to [their github](https://github.com/herobd/FUDGE) for installation and weights download
- replace `run.py` and `model/yolo_box_detector.py` scripts with our custom version from [this drive](https://drive.google.com/drive/folders/1K66A_z-x7cF9piHN_T8TWuJ8k9LOAm7Y?usp=sharing)

## Testing
### FUNSD

**Entity Labeling** reproduction:
```
python src/main.py -addG -addT -addE --model gcn --task elab --test --weights node.pt
```
- with Groun Truth: F1 Score: Macro 0.6921 - Micro 0.7851

- with Preprocessing (Detector and OCRs)

**Entity Linking** reproduction:
```
python src/main.py -addG -addT -addE --model edge --task elin --test --weights edge.pt
```
- with Groun Truth: F1 Classes: None 0.9961 - Pairs 0.5606

- with Preprocessing (Detector and OCRs)

### NAF
**Simple Subset**
```
python src/main.py -addG -addF -addE --model edge --task elin --src-data NAF --test -w naf.pt
```
- with Groun Truth: F1 Classes: None 0.9900 - Key-Value 0.4718 - SameEntity 0.2985
