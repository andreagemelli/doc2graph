import random
import spacy
import torch
import torchvision
from tqdm import tqdm
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as tvF
import numpy as np
import sys, os

from src.paths import CHECKPOINTS, FUDGE, ROOT
from src.models.unet import Unet
from src.data.amazing_utils import to_bin, transform_image
from src.data.amazing_utils import distance, get_histogram
from src.amazing_utils import get_config
sys.path.insert(0, os.path.join(ROOT, 'src/models/yolo/yolov5'))
from src.models.yolo.yolov5.models.common import DetectMultiBackend

# sys.path.append(os.path.join(ROOT,'FUDGE'))
# from FUDGE.run import detect_boxes

class FeatureBuilder():

    def __init__(self, d) -> None:
        """_summary_

        Args:
            d (_type_): _description_
        """
        self.cfg_preprocessing = get_config('preprocessing')
        self.device = d
        self.add_geom = self.cfg_preprocessing.FEATURES.add_geom
        self.add_embs = self.cfg_preprocessing.FEATURES.add_embs
        self.add_hist = self.cfg_preprocessing.FEATURES.add_hist
        self.add_visual = self.cfg_preprocessing.FEATURES.add_visual
        self.add_eweights = self.cfg_preprocessing.FEATURES.add_eweights
        self.add_fudge = self.cfg_preprocessing.FEATURES.add_fudge

        if self.add_embs:
            self.text_embedder = spacy.load('en_core_web_lg')
            # fasttext.util.download_model('en', if_exists='ignore')
            # self.text_embedder = fasttext.load_model('cc.en.300.bin')

        if self.add_visual:
            #self.visual_embedder = Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=1, classes=4)
            self.visual_embedder = torchvision.models.mobilenet_v3_large(pretrained=False)
            self.visual_embedder.classifier = torch.nn.Linear(960, 16)
            self.visual_embedder.load_state_dict(torch.load(CHECKPOINTS / 'mobilenet_v3_large.pth')['weights'])
            self.visual_embedder = self.visual_embedder.features
            self.visual_embedder = self.visual_embedder.to(d)

        if self.add_fudge:
            self.fudge = FUDGE / 'saved/NAF_detect_augR_staggerLighter.pth'
    
    def add_features(self, graphs, features):

        # rand_id = random.randint(0, len(graphs)-1)

        for id, g in enumerate(tqdm(graphs, desc='adding features:')):

            # positional features
            size = Image.open(features['paths'][id]).size
            sg = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
            sv = lambda r, f : [r[0]*f[1], r[1]*f[0], r[2]*f[1], r[3]*f[0]]
            feats = [[] for _ in range(len(features['boxs'][id]))]
            geom = [sg(box, size) for box in features['boxs'][id]]
            chunks = []
            
            # 'geometrical' features
            if self.add_geom:
                
                #boxs = torch.tensor([scale(box, size) for box in features['boxs'][id]]).to(self.device)
                #boxs = self.fourier_tf(boxs)
                [feats[idx].extend(sg(box, size)) for idx, box in enumerate(features['boxs'][id])]
                #[feats[idx].extend(box) for idx, box in enumerate(boxs)]
                chunks.append(4)
            
            if self.add_hist:
                # HISTOGRAM OF TEXT
                # [feats[idx].extend(hist) for idx, hist in enumerate(get_histogram(features['texts'][id]))]
                g.ndata['hist'] = torch.tensor([hist for hist in get_histogram(features['texts'][id])], dtype=torch.float32)
            
            # textual features
            if self.add_embs:
                # LANGUAGE MODEL (SPACY)
                [feats[idx].extend(self.text_embedder(features['texts'][id][idx]).vector) for idx, _ in enumerate(feats)]
                chunks.append(len(self.text_embedder(features['texts'][id][0]).vector))
                # LANGUAGE MODEL (FASTTEXT)
                # [feats[idx].extend(self.text_embedder.get_sentence_vector(features['texts'][id][idx])) for idx, _ in enumerate(feats)]
                # chunks.append(len(self.text_embedder.get_sentence_vector(features['texts'][id][0])))
            
            # visual features
            if self.add_visual:
                # https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html?highlight=roi
                img = Image.open(features['paths'][id]).convert('RGB')
                img = torchvision.transforms.functional.resize(img, size=(1000, 754))
                visual_emb = self.visual_embedder(tvF.to_tensor(img).unsqueeze_(0).to(self.device)) # output [batch, canali, dim1, dim2]
                factor = (1000/size[0], 754/size[1])
                bboxs = [torch.Tensor(sv(b, factor)) for b in features['boxs'][id]]
                bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
                # h = [torchvision.ops.roi_align(input=ve, boxes=bboxs, spatial_scale=1/ min(size[1] / ve.shape[2] , size[0] / ve.shape[3]), output_size=1) for ve in visual_emb[1:]]
                # h = torch.cat(h, dim=1)

                # yolo = DetectMultiBackend(CHECKPOINTS / 'yolo.pt', device=torch.device(self.device), dnn=False, data='/home/gemelli/projects/doc2graph/src/models/yolo/yolov5/data/coco128.yaml', fp16=False)
                # yolo = torch.nn.Sequential(*yolo.model.model[:10])
                # print(yolo)
                # transform = torchvision.transforms.Compose([
                #    torchvision.transforms.PILToTensor()
                # ])
                # img = (transform(img).float()).to(self.device)
                # img = img[None, :, :, :]
                # yolo_emb = yolo(img)
                # print(1000 / visual_emb.shape[2] , 754 / visual_emb.shape[3])
                h = torchvision.ops.roi_align(input=visual_emb, boxes=bboxs, spatial_scale = 1 / min(1000 / visual_emb.shape[2] , 754 / visual_emb.shape[3]), output_size=1)

                # VISUAL FEATURES (RESNET-IMAGENET)
                [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
                chunks.append(len(torch.flatten(h[0]).tolist()))
            
            # FUDGE visual features
            if self.add_fudge:
                # img = features['images'][id]
                full_path = features['paths'][id]
                # model = FUDGE / 'saved/NAF_detect_augR_staggerLighter.pth'
                model = FUDGE / 'saved/FUNSDLines_detect_augR_staggerLighter.pth'
                img_input = transform_image(full_path)

                _, visual_emb = detect_boxes(
                    img_input,
                    img_path = full_path,
                    output_path=ROOT,
                    include_threshold= 0.8,
                    model_checkpoint = model,
                    device='cuda:0',
                    detect = False)
                
                # visual_emb = torch.tensor(visual_emb).to(self.device)
                visual_emb = visual_emb.clone().detach().to(self.device)
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
                    angles.append(angle)
                
                m = max(distances)
                polar_coordinates = to_bin(distances, angles)
                g.edata['feat'] = polar_coordinates

            else:
                distances = ([0.0 for _ in range(g.number_of_edges())])
                m = 1

            g.ndata['geom'] = torch.tensor(geom, dtype=torch.float32)
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)

            distances = torch.tensor([(1-d/m) for d in distances], dtype=torch.float32)
            tresh_dist = torch.where(distances > 0.9, torch.full_like(distances, 0.1), torch.zeros_like(distances))
            g.edata['weights'] = tresh_dist

            norm = []
            num_nodes = len(features['boxs'][id]) - 1
            for n in range(num_nodes + 1):
                neigs = torch.count_nonzero(tresh_dist[n*num_nodes:(n+1)*num_nodes]).tolist()
                try: norm.append([1. / neigs])
                except: norm.append([1.])
            g.ndata['norm'] = torch.tensor(norm, dtype=torch.float32)

            #! DEBUG PURPOSES TO VISUALIZE RANDOM GRAPH IMAGE FROM DATASET
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

        return chunks, len(chunks)
    
    def fourier_tf(self, x):
        B_gauss = torch.randn((50, 4)).to(self.device) * 10
        x_proj = (2. * np.pi * x) @ B_gauss.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).tolist()
    
    def get_info(self):
        print(f"-> textual feats: {self.add_embs}\n-> visual feats: {self.add_visual}\n-> edge feats: {self.add_eweights}")

    
