from math import sqrt
import random

import numpy as np
from scipy.fft import dst
from src.data.amazing_utils import distance, get_histogram
from src.amazing_utils import get_config
import spacy
import torch
import torchvision
from torchvision import transforms
from src.paths import FUDGE, ROOT
import sys
import os
from src.data.amazing_utils import transform_image
sys.path.append(os.path.join(ROOT,'FUDGE'))
from FUDGE.run import detect_boxes
from tqdm import tqdm
from PIL import Image, ImageDraw

class FeatureBuilder():

    def __init__(self, d) -> None:
        """_summary_

        Args:
            d (_type_): _description_
        """
        #TODO add add_geom to pipeline from main
        self.cfg_preprocessing = get_config('preprocessing')
        self.add_geom = self.cfg_preprocessing.FEATURES.add_geom
        self.add_embs = self.cfg_preprocessing.FEATURES.add_embs
        self.add_visual = self.cfg_preprocessing.FEATURES.add_visual
        self.add_eweights = self.cfg_preprocessing.FEATURES.add_eweights
        self.device = d
        self.add_fudge = self.cfg_preprocessing.FEATURES.add_fudge

        if self.add_embs:
            self.text_embedder = spacy.load('en_core_web_lg')

        if self.add_visual: 
            self.convert_tensor = transforms.ToTensor()
            self.visual_embedder = torchvision.models.resnet18(pretrained=True)
            self.visual_embedder = torch.nn.Sequential(*(list(self.visual_embedder.children())[:-1])).to(self.device)
            self.visual_embedder.eval()

        if self.add_fudge:
            self.fudge = FUDGE / 'saved/NAF_detect_augR_staggerLighter.pth'
    
    def add_features(self, graphs, features):

        rand_id = random.randint(0, len(graphs)-1)

        for id, g in enumerate(tqdm(graphs, desc='adding features:')):

            # positional features
            size = Image.open(features['paths'][id]).size
            scale = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
            feats = [[] for _ in range(len(features['boxs'][id]))]
            chunks = []
            
            # 'geometrical' features
            if self.add_geom:
                [feats[idx].extend(scale(box, size)) for idx, box in enumerate(features['boxs'][id])]
                chunks.append(4)
            
            # textual features
            if self.add_embs:
                # HISTOGRAM OF TEXT
                [feats[idx].extend(hist) for idx, hist in enumerate(get_histogram(features['texts'][id]))]
                chunks.append(4)
                # LANGUAGE MODEL (SPACY)
                [feats[idx].extend(self.text_embedder(features['texts'][id][idx]).vector) for idx, _ in enumerate(feats)]
                chunks.append(len(self.text_embedder(features['texts'][id][0]).vector))
            
            # visual features
            if self.add_visual:
                # https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html?highlight=roi
                img = Image.open(features['paths'][id])
                img = self.convert_tensor(img).unsqueeze(dim=0).to(self.device)
                visual_emb = self.visual_embedder(img) # output [batch, canali, dim1, dim2]
                bboxs = [torch.Tensor(b) for b in features['boxs'][id]]
                bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
                scale = min(size[1] / visual_emb.shape[2] , size[0] / visual_emb.shape[3])
                #! output_size set for dimensionality and sapling_ratio at random.
                h = torchvision.ops.roi_align(input=visual_emb, boxes=bboxs, spatial_scale=1/scale, output_size=1, sampling_ratio=3)

                # VISUAL FEATURES (RESNET-IMAGENET)
                [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
                chunks.append(len(torch.flatten(h[0]).tolist()))
            
            # FUDGE visual features
            if self.add_fudge:
                # img = features['images'][id]
                full_path = features['paths'][id]
                model = FUDGE / 'saved/NAF_detect_augR_staggerLighter.pth'
                img_input = transform_image(full_path)

                _, visual_emb = detect_boxes(
                    img_input,
                    img_path = full_path,
                    output_path=ROOT,
                    include_threshold= 0.8,
                    model_checkpoint = model,
                    device='cuda:0',
                    detect = False)
                
                visual_emb = torch.tensor(visual_emb).to(self.device)
                # visual_emb = torch.tensor(visual_emb.clone().detach()).to(self.device)
                
                bboxs = [torch.Tensor(b) for b in features['boxs'][id]]
                bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
                scale = min(size[1] / visual_emb.shape[2] , size[0] / visual_emb.shape[3])
                #! output_size set for dimensionality and sapling_ratio at random.
                h = torchvision.ops.roi_align(input=visual_emb, boxes=bboxs, spatial_scale=1/scale, output_size=1, sampling_ratio=3)

                [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
                chunks.append(len(torch.flatten(h[0]).tolist()))
            
            if self.add_eweights:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                distances = []
                angles = []

                # TODO CHOOSE WHICH DISTANCE NORMALIZATION TO APPLY
                #! with fully connected simply normalized with max distance between distances
                # m = sqrt((size[0]*size[0] + size[1]*size[1]))
                # parable = lambda x : (-x+1)**4
                
                for pair in zip(srcs, dsts):
                    dist, angle = distance(features['boxs'][id][pair[0]], features['boxs'][id][pair[1]])
                    
                    distances.append(dist)
                    angles.append(angle/360)
                
                m = max(distances)
                g.edata['angl'] = torch.tensor(angles, dtype=torch.float32)

            else:
                distances = [0.0 for _ in range(g.number_of_edges())]
                m = 1

            # add features to graph
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
            g.edata['dist'] = torch.tensor([(1-d/m) for d in distances], dtype=torch.float32)

            if False:
                if id == rand_id and self.add_eweights:
                    print("\n\n### EXAMPLE ###")

                    img_path = features['paths'][id]
                    img = Image.open(img_path).convert('RGB')
                    draw = ImageDraw.Draw(img)

                    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
                    select = [random.randint(0, len(srcs)) for _ in range(10)]
                    for p, pair in enumerate(zip(srcs, dsts)):
                        if p in select:
                            sc = center(features['boxs'][id][pair[0]])
                            ec = center(features['boxs'][id][pair[1]])
                            draw.line((sc, ec), fill='grey', width=3)
                            middle_point = ((sc[0] + ec[0])/2,(sc[1] + ec[1])/2)
                            draw.text(middle_point, str(angles[p]), fill='black')
                            draw.rectangle(features['boxs'][id][pair[0]], fill='red')
                            draw.rectangle(features['boxs'][id][pair[1]], fill='blue')
                    
                    img.save(f'esempi/FUNSD/edges.png')

        return chunks
    
    def get_info(self):
        print(f"-> textual feats: {self.add_embs}\n-> visual feats: {self.add_visual}\n-> edge feats: {self.add_eweights}")

    