import random
from typing import Tuple
import spacy
import torch
import torchvision
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as tvF

from src.paths import CHECKPOINTS
from src.models.unet import Unet
from src.data.utils import to_bin, to_position
from src.data.utils import polar
from src.utils import get_config
import src.globals as glb

class FeatureBuilder():

    def __init__(self):
        """FeatureBuilder constructor
        """
        self.cfg_preprocessing = get_config('preprocessing')
        self.device = glb.DEVICE
        self.add_layout = self.cfg_preprocessing.FEATURES.add_layout
        self.add_text = self.cfg_preprocessing.FEATURES.add_text
        self.add_visual = self.cfg_preprocessing.FEATURES.add_visual
        self.add_efeat = self.cfg_preprocessing.FEATURES.add_efeat
        self.num_polar_bins = self.cfg_preprocessing.FEATURES.num_polar_bins

        if self.add_text:
            self.text_embedder = spacy.load('en_core_web_lg')

        # if self.add_visual:
            # self.visual_embedder = Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=1, classes=4)
            # self.visual_embedder.load_state_dict(torch.load(CHECKPOINTS / 'backbone_unet.pth')['weights'])
            # self.visual_embedder = self.visual_embedder.encoder
            # self.visual_embedder.to(self.device)
        
        self.sg = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1], (rect[2] - rect[0])/s[0], (rect[3] - rect[1])/s[1]] # scaling by img width and height
    
    def add_features(self, graphs : list, features : list) -> Tuple[list, int]:
        """ Add features to provided graphs

        Args:
            graphs (list) : list of DGLGraphs
            features (list) : list of features "sources", like text, positions and images
        
        Returns:
            chunks list and its lenght
        """

        rand_id = random.randint(0, len(graphs) - 1)
        for id, g in enumerate(tqdm(graphs, desc='adding features')):

            # positional features
            size = Image.open(features['paths'][id]).size
            feats = [[] for _ in range(len(features['boxs'][id]))]
            chunks = []

            # 'geometrical' features
            if self.add_layout:
                
                # OLD DOC2GRAPH
                [feats[idx].extend(self.sg(box, size)) for idx, box in enumerate(features['boxs'][id])]
                # [feats[idx].extend([box[0], box[1] - 1, box[2], box[3] - 1, box[2] - box[0], box[3] - box[1]]) for idx, box in enumerate(features['boxs'][id])]
                chunks.append(len(feats[0]))
            
            # textual features
            if self.add_text:
                
                # LANGUAGE MODEL (SPACY)
                [feats[idx].extend(self.text_embedder(features['texts'][id][idx]).vector) for idx, _ in enumerate(feats)]
                chunks.append(len(self.text_embedder(features['texts'][id][0]).vector))
            
            # visual features
            if self.add_visual:
                # img = Image.open(features['paths'][id])
                # # ti = tvF.to_tensor(img).unsqueeze_(0)
                # visual_emb = self.visual_embedder(tvF.to_tensor(img).unsqueeze_(0).to(self.device)) # output [batch, channels, dim1, dim2]
                # bboxs = [torch.Tensor(b) for b in features['boxs'][id]]
                # bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
                # h = [torchvision.ops.roi_align(input=ve, boxes=bboxs, spatial_scale=1/ min(size[1] / ve.shape[2] , size[0] / ve.shape[3]), output_size=1) for ve in visual_emb[1:]]
                # h = torch.cat(h, dim=1)

                # # VISUAL FEATURES (RESNET-IMAGENET)
                # [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
                chunks.append(1024)
        
            if self.add_efeat:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                distances = []
                angles = []

                # TODO CHOOSE WHICH DISTANCE NORMALIZATION TO APPLY
                #! with fully connected simply normalized with max distance between distances
                # m = sqrt((size[0]*size[0] + size[1]*size[1]))
                # parable = lambda x : (-x+1)**4
                
                for pair in zip(srcs, dsts):
                    #! they should be inverted polar(dst, src)!
                    dist, angle = polar(features['boxs'][id][pair[1]], features['boxs'][id][pair[0]])
                    distances.append(dist)
                    angles.append(angle)
                
                m = max(distances)
                polar_coordinates = to_bin(distances, angles, self.num_polar_bins)
                position = to_position(angles)
                g.edata['feat'] = polar_coordinates
                g.edata['position'] = torch.tensor(position, dtype=torch.float32)

            else:
                distances = ([0.0 for _ in range(g.number_of_edges())])
                m = 1

            g.ndata['layout'] = torch.tensor(features['boxs'][id], dtype=torch.float32)
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)

            norm_distances = torch.tensor([(1-d/m) for d in distances], dtype=torch.float32)
            tresh_dist = torch.where(norm_distances > 0.9, torch.full_like(norm_distances, 0.1), torch.zeros_like(norm_distances))
            g.edata['dist'] = tresh_dist

            norm = []
            num_nodes = len(features['boxs'][id]) - 1
            for n in range(num_nodes + 1):
                neigs = torch.count_nonzero(tresh_dist[n*num_nodes:(n+1)*num_nodes]).tolist()
                try: norm.append([1. / neigs])
                except: norm.append([1.])
            g.ndata['norm'] = torch.tensor(norm, dtype=torch.float32)

            #! DEBUG PURPOSES TO VISUALIZE RANDOM GRAPH IMAGE FROM DATASET
            if False:
                center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

                if id == rand_id and self.add_efeat:
                    img_path = features['paths'][id]
                    img = Image.open(img_path).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    for h in range(0, g.num_nodes()):
                        draw.rectangle(features['boxs'][id][h], outline='gray', width=3)

                    select = random.randint(0, g.num_nodes() - 1)
                    start = select*g.num_nodes() - select
                    end = (select + 1)*g.num_nodes() - 1 - select
                    neighbors = [idx for idx, value in enumerate(tresh_dist[start : end]) if value != 0]
                    pos = []
                    edges = []
                    for n, ne in enumerate(neighbors):
                        if ne >= select:
                            neighbors[n] += 1

                        s = neighbors[n]*g.num_nodes() - neighbors[n]
                        if select >= neighbors[n]: 
                            pos.append(angles[s+select-1])
                            edges.append(s+select-1)
                        else: 
                            pos.append(angles[s+select])
                            edges.append(s+select)

                    central_node = features['boxs'][id][select]
                    draw.rectangle(central_node, outline='red', width=2)
                    neighbor_nodes = [features['boxs'][id][n] for n in neighbors]

                    sc = center(central_node)
                    radius = int(m*0.1)
                    draw.rectangle((central_node[0] - radius, central_node[1] - radius, central_node[2] + radius, central_node[3] + radius), outline='red', width=2)
                    draw.line((sc[0] - radius*2, sc[1] - radius*2, sc[0] + radius*2, sc[1] + radius*2), fill='red', width=2)
                    draw.line((sc[0] + radius*2, sc[1] - radius*2, sc[0] - radius*2, sc[1] + radius*2), fill='red', width=2)

                    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 12)
                    for n, ne in enumerate(neighbor_nodes):
                        ec = center(ne)
                        draw.line((sc, ec), fill='green', width=2)
                        if position[edges[n]] == 0:
                            draw.rectangle(ne, outline='blue', width=3)
                        elif position[edges[n]] == 1:
                            draw.rectangle(ne, outline='brown', width=3)
                        elif position[edges[n]] == 2:
                            draw.rectangle(ne, outline='violet', width=3)
                        else:
                            draw.rectangle(ne, outline='pink', width=3)

                        middle_point = ((sc[0] + ec[0])/2,(sc[1] + ec[1])/2)
                        # draw.text(middle_point, str(pos[n]), fill='black')

                        bbox = draw.textbbox(middle_point, str(pos[n]), font=font)
                        draw.rectangle(bbox, fill="green")
                        draw.text(middle_point, str(pos[n]), font=font, fill="white")
                        
                    
                    img.save(f'edges.png')

        return chunks, len(chunks)
    
    def get_info(self):
        print(f"-> textual feats: {self.add_text}\n-> visual feats: {self.add_visual}\n-> edge feats: {self.add_efeat}")

    
