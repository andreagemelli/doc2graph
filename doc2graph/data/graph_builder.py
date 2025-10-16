import json
import os
from PIL import Image, ImageDraw
from typing import Tuple
import torch
import dgl
import random
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import easyocr

from src.data.preprocessing import load_predictions
from src.data.utils import polar
from src.paths import DATA, FUNSD_TEST
from src.utils import get_config


class GraphBuilder():

    def __init__(self):
        self.cfg_preprocessing = get_config('preprocessing')
        self.edge_type = self.cfg_preprocessing.GRAPHS.edge_type
        self.data_type = self.cfg_preprocessing.GRAPHS.data_type
        self.node_granularity = self.cfg_preprocessing.GRAPHS.node_granularity
        random.seed = 42
        return
    
    def get_graph(self, src_path : str, src_data : str) -> Tuple[list, list, list, list]:
        """ Given the source, it returns a graph

        Args:
            src_path (str) : path to source data
            src_data (str) : either FUNSD, PAU or CUSTOM
        
        Returns:
            tuple (lists) : graphs, nodes and edge labels, features
        """
        
        if src_data == 'FUNSD':
            return self.__fromFUNSD(src_path)
        elif src_data == 'PAU':
            return self.__fromPAU(src_path)
        elif src_data == 'CUSTOM':
            if self.data_type == 'img':
                return self.__fromIMG(src_path)
            elif self.data_type == 'pdf':
                return self.__fromPDF()
            else:
                raise Exception('GraphBuilder exception: data type invalid. Choose from ["img", "pdf"]')
        else:
            raise Exception('GraphBuilder exception: source data invalid. Choose from ["FUNSD", "PAU", "CUSTOM"]')
    
    def balance_edges(self, g : dgl.DGLGraph, cls=None ) -> dgl.DGLGraph:
        """ if cls (class) is not None, but an integer instead, balance that class to be equal to the sum of the other classes

        Args:
            g (DGLGraph) : a DGL graph
            cls (int) : class number, if any
        
        Returns:
            g (DGLGraph) : the new balanced graph
        """
        
        edge_targets = g.edata['label']
        u, v = g.all_edges(form='uv')
        edges_list = list()
        for e in zip(u.tolist(), v.tolist()):
            edges_list.append([e[0], e[1]])

        if type(cls) is int:
            to_remove = (edge_targets == cls)
            indices_to_remove = to_remove.nonzero().flatten().tolist()

            for _ in range(int((edge_targets != cls).sum()/2)):
                indeces_to_save = [random.choice(indices_to_remove)]
                edge = edges_list[indeces_to_save[0]]

                for index in sorted(indeces_to_save, reverse=True):
                    del indices_to_remove[indices_to_remove.index(index)]

            indices_to_remove = torch.flatten(torch.tensor(indices_to_remove, dtype=torch.int32))
            g = dgl.remove_edges(g, indices_to_remove)
            return g
            
        else:
            raise Exception("Select a class to balance (an integer ranging from 0 to num_edge_classes).")
    
    def get_info(self):
        """ returns graph information
        """
        print(f"-> edge type: {self.edge_type}")

    def fully_connected(self, ids : list) -> Tuple[list, list]:
        """ create fully connected graph

        Args:
            ids (list) : list of node indices
        
        Returns:
            u, v (lists) : lists of indices
        """
        u, v = list(), list()
        for id in ids:
            u.extend([id for i in range(len(ids)) if i != id])
            v.extend([i for i in range(len(ids)) if i != id])
        return u, v
    
    def knn_connection(self, size : tuple, bboxs : list, k = 10) -> Tuple[list, list]:
        """ Given a list of bounding boxes, find for each of them their k nearest ones.

        Args:
            size (tuple) : width and height of the image
            bboxs (list) : list of bounding box coordinates
            k (int) : k of the knn algorithm
        
        Returns:
            u, v (lists) : lists of indices
        """

        edges = []
        width, height = size[0], size[1]
        
        # creating projections
        vertical_projections = [[] for i in range(width)]
        horizontal_projections = [[] for i in range(height)]
        for node_index, bbox in enumerate(bboxs):
            for hp in range(bbox[0], bbox[2]):
                if hp >= width: hp = width - 1
                vertical_projections[hp].append(node_index)
            for vp in range(bbox[1], bbox[3]):
                if vp >= height: vp = height - 1
                horizontal_projections[vp].append(node_index)
        
        def bound(a, ori=''):
            if a < 0 : return 0
            elif ori == 'h' and a > height: return height
            elif ori == 'w' and a > width: return width
            else: return a

        for node_index, node_bbox in enumerate(bboxs):
            neighbors = [] # collect list of neighbors
            window_multiplier = 2 # how much to look around bbox
            wider = (node_bbox[2] - node_bbox[0]) > (node_bbox[3] - node_bbox[1]) # if bbox wider than taller
            
            ### finding neighbors ###
            while(len(neighbors) < k and window_multiplier < 100): # keep enlarging the window until at least k bboxs are found or window too big
                vertical_bboxs = []
                horizontal_bboxs = []
                neighbors = []
                
                if wider:
                    h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier/4)
                    v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier)
                else:
                    h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier)
                    v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier/4)
                
                window = [bound(node_bbox[0] - h_offset),
                        bound(node_bbox[1] - v_offset),
                        bound(node_bbox[2] + h_offset, 'w'),
                        bound(node_bbox[3] + v_offset, 'h')] 
                
                [vertical_bboxs.extend(d) for d in vertical_projections[window[0]:window[2]]]
                [horizontal_bboxs.extend(d) for d in horizontal_projections[window[1]:window[3]]]
                
                for v in set(vertical_bboxs):
                    for h in set(horizontal_bboxs):
                        if v == h: neighbors.append(v)
                
                window_multiplier += 1 # enlarge the window
            
            ### finding k nearest neighbors ###
            neighbors = list(set(neighbors))
            if node_index in neighbors:
                neighbors.remove(node_index)
            neighbors_distances = [polar(node_bbox, bboxs[n])[0] for n in neighbors]
            for sd_num, sd_idx in enumerate(np.argsort(neighbors_distances)):
                if sd_num < k:
                    if [node_index, neighbors[sd_idx]] not in edges and [neighbors[sd_idx], node_index] not in edges:
                        edges.append([neighbors[sd_idx], node_index])
                        edges.append([node_index, neighbors[sd_idx]])
                else: break

        return [e[0] for e in edges], [e[1] for e in edges]
    
    def __fromIMG(self, paths : list):
        
        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'paths': paths, 'texts': [], 'boxs': []}

        for path in paths:
            reader = easyocr.Reader(['en']) #! TODO: in the future, handle multilanguage!
            result = reader.readtext(path, paragraph=True)
            img = Image.open(path).convert('RGB')
            draw = ImageDraw.Draw(img)
            boxs, texts = list(), list()

            for r in result:
                box = [int(r[0][0][0]), int(r[0][0][1]), int(r[0][2][0]), int(r[0][2][1])]
                draw.rectangle(box, outline='red', width=3)
                boxs.append(box)
                texts.append(r[1])

            features['boxs'].append(boxs)
            features['texts'].append(texts)
            img.save('prova.png')

            if self.edge_type == 'fully':
                u, v = self.fully_connected(range(len(boxs)))
            elif self.edge_type == 'knn': 
                u,v = self.knn_connection(Image.open(path).size, boxs)
            else:
                raise Exception('Other edge types still under development.')

            g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(boxs), idtype=torch.int32)
            graphs.append(g)

        return graphs, node_labels, edge_labels, features
    
    def __fromPDF():
        #TODO: dev from PDF import of graphs
        return

    def __fromPAU(self, src: str) -> Tuple[list, list, list, list]:
        """ build graphs from Pau Riba's dataset

        Args:
            src (str) : path to where data is stored
        
        Returns:
            tuple (lists) : graphs, nodes and edge labels, features
        """

        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'paths': [], 'texts': [], 'boxs': []}

        for image in tqdm(os.listdir(src), desc='Creating graphs'):
            if not image.endswith('tif'): continue
            
            img_name = image.split('.')[0]
            file_gt = img_name + '_gt.xml'
            file_ocr = img_name + '_ocr.xml'
            
            if not os.path.isfile(os.path.join(src, file_gt)) or not os.path.isfile(os.path.join(src, file_ocr)): continue
            features['paths'].append(os.path.join(src, image))

            # DOCUMENT REGIONS
            root = ET.parse(os.path.join(src, file_gt)).getroot()
            regions = []
            for parent in root:
                if parent.tag.split("}")[1] == 'Page':
                    for child in parent:
                        region_label = child[0].attrib['value']
                        region_bbox = [int(child[1].attrib['points'].split(" ")[0].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[1].split(",")[1].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[2].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[3].split(",")[1].split(".")[0])]
                        regions.append([region_label, region_bbox])

            # DOCUMENT TOKENS
            root = ET.parse(os.path.join(src, file_ocr)).getroot()
            tokens_bbox = []
            tokens_text = []
            nl = []
            center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
            for parent in root:
                if parent.tag.split("}")[1] == 'Page':
                    for child in parent:
                        if child.tag.split("}")[1] == 'TextRegion':
                            for elem in child:
                                if elem.tag.split("}")[1] == 'TextLine':
                                    for word in elem:
                                        if word.tag.split("}")[1] == 'Word':
                                            word_bbox = [int(word[0].attrib['points'].split(" ")[0].split(",")[0].split(".")[0]),
                                                        int(word[0].attrib['points'].split(" ")[1].split(",")[1].split(".")[0]),
                                                        int(word[0].attrib['points'].split(" ")[2].split(",")[0].split(".")[0]),
                                                        int(word[0].attrib['points'].split(" ")[3].split(",")[1].split(".")[0])]
                                            word_text = word[1][0].text
                                            c = center(word_bbox)
                                            for reg in regions:
                                                r = reg[1]
                                                if r[0] < c[0] < r[2] and r[1] < c[1] < r[3]:
                                                    word_label = reg[0]
                                                    break
                                            tokens_bbox.append(word_bbox)
                                            tokens_text.append(word_text)
                                            nl.append(word_label)
            
            features['boxs'].append(tokens_bbox)
            features['texts'].append(tokens_text)
            node_labels.append(nl)

            # getting edges
            if self.edge_type == 'fully':
                u, v = self.fully_connected(range(len(tokens_bbox)))
            elif self.edge_type == 'knn': 
                u,v = self.knn_connection(Image.open(os.path.join(src, image)).size, tokens_bbox)
            else:
                raise Exception('Other edge types still under development.')
            
            el = list()
            for e in zip(u, v):
                if (nl[e[0]] == nl[e[1]]) and (nl[e[0]] == 'positions' or nl[e[0]] == 'total'):
                    el.append('table')
                else: el.append('none')
            edge_labels.append(el)

            g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(tokens_bbox), idtype=torch.int32)
            graphs.append(g)
        
        return graphs, node_labels, edge_labels, features

    def __fromFUNSD(self, src : str) -> Tuple[list, list, list, list]:
        """Parsing FUNSD annotation files

        Args:
            src (str) : path to where data is stored
        
        Returns:
            tuple (lists) : graphs, nodes and edge labels, features
        """

        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'paths': [], 'texts': [], 'boxs': []}
        # justOne = random.choice(os.listdir(os.path.join(src, 'adjusted_annotations'))).split(".")[0]
        
        if self.node_granularity == 'gt':
            for file in tqdm(os.listdir(os.path.join(src, 'adjusted_annotations')), desc='Creating graphs - GT'):
            
                img_name = f'{file.split(".")[0]}.png'
                img_path = os.path.join(src, 'images', img_name)
                features['paths'].append(img_path)

                with open(os.path.join(src, 'adjusted_annotations', file), 'r') as f:
                    form = json.load(f)['form']

                # getting infos
                boxs, texts, ids, nl = list(), list(), list(), list()
                pair_labels = list()

                for elem in form:
                    boxs.append(elem['box'])
                    texts.append(elem['text'])
                    nl.append(elem['label'])
                    ids.append(elem['id'])
                    [pair_labels.append(pair) for pair in elem['linking']]
                
                for p, pair in enumerate(pair_labels):
                    pair_labels[p] = [ids.index(pair[0]), ids.index(pair[1])]
                
                node_labels.append(nl)
                features['texts'].append(texts)
                features['boxs'].append(boxs)
                
                # getting edges
                if self.edge_type == 'fully':
                    u, v = self.fully_connected(range(len(boxs)))
                elif self.edge_type == 'knn': 
                    u,v = self.knn_connection(Image.open(img_path).size, boxs)
                else:
                    raise Exception('GraphBuilder exception: Other edge types still under development.')
                
                el = list()
                for e in zip(u, v):
                    edge = [e[0], e[1]]
                    if edge in pair_labels: el.append('pair')
                    else: el.append('none')
                edge_labels.append(el)

                # creating graph
                g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(boxs), idtype=torch.int32)
                graphs.append(g)

            #! DEBUG PURPOSES TO VISUALIZE RANDOM GRAPH IMAGE FROM DATASET
            if False:
                if justOne == file.split(".")[0]:
                    print("\n\n### EXAMPLE ###")
                    print("Savin example:", img_name)

                    edge_unique_labels = np.unique(el)
                    g.edata['label'] = torch.tensor([np.where(target == edge_unique_labels)[0][0] for target in el])

                    g = self.balance_edges(g, 3, int(np.where('none' == edge_unique_labels)[0][0]))

                    img_removed = Image.open(img_path).convert('RGB')
                    draw_removed = ImageDraw.Draw(img_removed)

                    for b, box in enumerate(boxs):
                        if nl[b] == 'header':
                            color = 'yellow'
                        elif nl[b] == 'question':
                            color = 'blue'
                        elif nl[b] == 'answer':
                            color = 'green'
                        else:
                            color = 'gray'
                        draw_removed.rectangle(box, outline=color, width=3)

                    u, v = g.all_edges()
                    labels = g.edata['label'].tolist()
                    u, v = u.tolist(), v.tolist()

                    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

                    num_pair = 0
                    num_none = 0

                    for p, pair in enumerate(zip(u,v)):
                        sc = center(boxs[pair[0]])
                        ec = center(boxs[pair[1]])
                        if labels[p] == int(np.where('pair' == edge_unique_labels)[0][0]): 
                            num_pair += 1
                            color = 'violet'
                            draw_removed.ellipse([(sc[0]-4,sc[1]-4), (sc[0]+4,sc[1]+4)], fill = 'green', outline='black')
                            draw_removed.ellipse([(ec[0]-4,ec[1]-4), (ec[0]+4,ec[1]+4)], fill = 'red', outline='black')
                        else: 
                            num_none += 1
                            color='gray'
                        draw_removed.line((sc,ec), fill=color, width=3)
                    
                    print("Balanced Links: None {} | Key-Value {}".format(num_none, num_pair))
                    img_removed.save(f'esempi/FUNSD/{img_name}_removed_graph.png')

        elif self.node_granularity == 'yolo':
            path_preds = os.path.join(src, 'yolo_bbox')
            path_images = os.path.join(src, 'images')
            path_gts = os.path.join(src, 'adjusted_annotations')
            all_paths, all_preds, all_links, all_labels, all_texts = load_predictions(path_preds, path_gts, path_images)
            for f, img_path in enumerate(tqdm(all_paths, desc='Creating graphs - YOLO')):
            
                features['paths'].append(img_path)
                features['boxs'].append(all_preds[f])
                features['texts'].append(all_texts[f])
                node_labels.append(all_labels[f])
                pair_labels = all_links[f]

                # getting edges
                if self.edge_type == 'fully':
                    u, v = self.fully_connected(range(len(features['boxs'][f])))
                elif self.edge_type == 'knn': 
                    u,v = self.knn_connection(Image.open(img_path).size, features['boxs'][f])
                else:
                    raise Exception('GraphBuilder exception: Other edge types still under development.')
                
                el = list()
                for e in zip(u, v):
                    edge = [e[0], e[1]]
                    if edge in pair_labels: el.append('pair')
                    else: el.append('none')
                edge_labels.append(el)

                # creating graph
                g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(features['boxs'][f]), idtype=torch.int32)
                graphs.append(g)
        else:
            #TODO develop OCR too
            raise Exception('GraphBuilder Exception: only \'gt\' or \'yolo\' available for FUNSD.')


        return graphs, node_labels, edge_labels, features