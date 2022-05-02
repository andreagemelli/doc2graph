import json
import os
from PIL import Image
import spacy
import torch
import dgl

embedder = spacy.load('en_core_web_sm')

def fully_connected(ids : list):
    # TODO add distance to edges
    u, v = list(), list()
    for id in ids:
        u.extend([id for i in range(len(ids))])
        v.extend([i for i in range(len(ids))])
    return torch.tensor(u), torch.tensor(v)

def fromNAT():
    #TODO
    return

def fromFUNSD(src : str, add_embs : bool):
    """ Parsing FUNSD annotation json files
    """
    graphs = []
    labels = []
    for file in os.listdir(os.path.join(src, 'annotations')):

        size = Image.open(os.path.join(src, 'images', f'{file.split(".")[0]}.png')).size

        with open(os.path.join(src, 'annotations', file), 'r') as f:
            form = json.load(f)['form']

        # getting infos
        boxs, texts, g_labels = list(), list(), list()

        for elem in form:
            if elem['text']:
                boxs.append(elem['box'])
                texts.append(elem['text'])
                g_labels.append(elem['label'])
        
        # getting edges
        assert len(boxs) == len(texts) == len(g_labels)
        node_ids = range(len(boxs))
        u, v = fully_connected(node_ids)

        # getting node features
        scale = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
        feats = [scale(box, size) for box in boxs]
        if add_embs:
            [feat.extend(embedder(texts[idx]).vector) for idx, feat in enumerate(feats)]

        # creating graph
        g = dgl.graph((u, v), num_nodes=len(boxs), idtype=torch.int32)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)

        graphs.append(g), labels.append(g_labels)

    return graphs, labels