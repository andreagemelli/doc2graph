from src.data.utils import distance
from src.utils import get_config
import spacy
import torch

class FeatureBuilder():

    def __init__(self) -> None:
        self.cfg_preprocessing = get_config('preprocessing')
        self.add_embs = self.cfg_preprocessing.FEATURES.add_embs
        if self.add_embs: self.embedder = spacy.load('en_core_web_sm')
        self.add_visual = self.cfg_preprocessing.FEATURES.add_visual
        # TODO: add visual embedder like ViT or ResNet or ..
        self.add_eweights = self.cfg_preprocessing.FEATURES.add_eweights
    
    def add_features(self, graphs, features):
        for id, g in enumerate(graphs):
            # positional features
            size = features['images'][id].size
            scale = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
            feats = [scale(box, size) for box in features['boxs'][id]]

            # textual features
            if self.add_embs:
                [feats[idx].extend(self.embedder(features['texts'][id][idx]).vector) for idx, _ in enumerate(feats)]
            
            if self.add_visual:
                raise Exception("Adding visual features still under developments")
            
            if self.add_eweights:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                distances = []
                
                for i, src in enumerate(srcs):
                    distances.append(distance(features['boxs'][id][src], features['boxs'][id][dsts[i]]))
                
                m = max(distances)
                distances = [(1 - d/m) for d in distances]
                
                g.edata['feat'] = torch.tensor(distances, dtype=torch.float32)

            # add features to graph
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
    
    def get_info(self):
        print(f"-> textual feats: {self.add_embs}\n-> visual feats: {self.add_visual}\n-> edge feats: {self.add_eweights}")

    