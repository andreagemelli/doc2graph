from src.data.amazing_utils import distance, get_histogram
from src.training.amazing_utils import get_device
from src.amazing_utils import get_config
import spacy
import torch
import torchvision
from torchvision import transforms

class FeatureBuilder():

    def __init__(self, d) -> None:
        """_summary_

        Args:
            d (_type_): _description_
        """
        self.cfg_preprocessing = get_config('preprocessing')
        self.add_embs = self.cfg_preprocessing.FEATURES.add_embs
        self.add_visual = self.cfg_preprocessing.FEATURES.add_visual
        self.add_eweights = self.cfg_preprocessing.FEATURES.add_eweights
        self.device = d
        if self.add_embs: 
            # self.text_embedder = spacy.load('en_core_web_sm')
            self.text_embedder = spacy.load('en_core_web_lg')
        if self.add_visual: 
            self.convert_tensor = transforms.ToTensor()
            self.visual_embedder = torchvision.models.resnet18(pretrained=True)
            self.visual_embedder = torch.nn.Sequential(*(list(self.visual_embedder.children())[:-1])).to(self.device)
            self.visual_embedder.eval()
    
    def add_features(self, graphs, features):
        for id, g in enumerate(graphs):
            # positional features
            size = features['images'][id].size
            # filename = features['images'][id].filename
            scale = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
            feats = [scale(box, size) for box in features['boxs'][id]]
            
            # simple embedding of the content
            [feats[idx].extend(hist) for idx, hist in enumerate(get_histogram(features['texts'][id]))]

            # textual features
            if self.add_embs:
                [feats[idx].extend(self.text_embedder(features['texts'][id][idx]).vector) for idx, _ in enumerate(feats)]
            
            if self.add_visual:
                # https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html?highlight=roi
                img = features['images'][id]
                img = self.convert_tensor(img).unsqueeze(dim=0).to(self.device)
                visual_emb = self.visual_embedder(img) # output [batch, canali, dim1, dim2]
                bboxs = [torch.Tensor(b) for b in features['boxs'][id]]
                bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
                scale = min(size[1] / visual_emb.shape[2] , size[0] / visual_emb.shape[3])
                #! output_size set for dimensionality and sapling_ratio at random.
                h = torchvision.ops.roi_align(input=visual_emb, boxes=bboxs, spatial_scale=1/scale, output_size=1, sampling_ratio=3)
                [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
            
            if self.add_eweights:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                distances = []
                
                for i, src in enumerate(srcs):
                    distances.append(distance(features['boxs'][id][src], features['boxs'][id][dsts[i]]))
                
                m = max(distances)
                distances = [(1 - d/m) for d in distances]

            else:
                distances = [1.0 for d in range(g.number_of_edges())]

            # add features to graph
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
            g.edata['feat'] = torch.tensor(distances, dtype=torch.float32)
    
    def get_info(self):
        print(f"-> textual feats: {self.add_embs}\n-> visual feats: {self.add_visual}\n-> edge feats: {self.add_eweights}")

    