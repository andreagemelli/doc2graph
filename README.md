# Documents 2 Graph
- [Documents 2 Graph](#documents-2-graph)
    - [Info](#info)
    - [Install](#install)
    - [Results](#results)
    - [Usage](#usage)

### Info
Library to convert documents to graphs and perform several tasks on different datasets, like KIE on FUNSD.

### Install
```
pip install .
```

### Results

### Usage
1. To train a **GCN** model for **Entity Labeling** on FUNSD (using CPU):
```
python src/main.py
```

2. To test a trained **GCN** model for **Entity Labeling** on FUNSD (using GPU):
```
python src/main.py --gpu 0 --config gcn --test -w output/weights/gcn-1651069965.pt
```