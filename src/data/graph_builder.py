import json
import os
from PIL import Image, ImageDraw
from src.data.amazing_utils import distance, organize_naf
from src.data.preprocessing import load_predictions
from src.paths import NAF
from src.amazing_utils import get_config
import torch
import dgl
import random
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET


class GraphBuilder():

    def __init__(self) -> None:
        self.cfg_preprocessing = get_config('preprocessing')
        self.edge_type = self.cfg_preprocessing.GRAPHS.edge_type
        self.data_type = self.cfg_preprocessing.GRAPHS.data_type
        random.seed = 42
        return
    
    def get_graph(self, src_path, src_data):
        
        if src_data == 'FUNSD':
            return self.__fromFUNSD(src_path)
        elif src_data == 'PAU':
            return self.__fromPAU(src_path)
        elif src_data == 'NAF':
            return self.__fromNAF(src_path)
        elif src_data == 'CUSTOM':
            if self.data_type == 'img':
                return self.__fromIMG()
            elif self.data_type == 'pdf':
                return self.__fromPDF()
            else:
                raise Exception('GraphBuilder exception: data type invalid. Choose from ["img", "pdf"]')
        else:
            raise Exception('GraphBuilder exception: source data invalid. Choose from ["FUNSD", "NAF", "CUSTOM"]')
    
    def balance_edges(self, g, num_classes, cls=None):
        # if cls is not None, but an integer inster, balance that class to be equal to the sum of the rest
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
                # indeces_to_save.append(edges_list.index([edge[1], edge[0]]))

                for index in sorted(indeces_to_save, reverse=True):
                    del indices_to_remove[indices_to_remove.index(index)]

            indices_to_remove = torch.flatten(torch.tensor(indices_to_remove, dtype=torch.int32))
            g = dgl.remove_edges(g, indices_to_remove)
            return g
            
        else:
            raise Exception("Select a class to balance (an integer ranging from 0 to num_edge_classes).")
    
    def get_info(self):
        print(f"-> edge type: {self.edge_type}")

    def __fully_connected(self, ids : list):
        u, v = list(), list()
        for id in ids:
            u.extend([id for i in range(len(ids)) if i != id])
            v.extend([i for i in range(len(ids)) if i != id])
        return u, v
    
    def __knn(self, size, bboxs):
        """ Given a list of bounding boxes, find for each of them their k nearest ones.
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

        # k = self.config.PREPROCESS.k
        k = 10

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
            neighbors_distances = [distance(node_bbox, bboxs[n])[0] for n in neighbors]
            for sd_num, sd_idx in enumerate(np.argsort(neighbors_distances)):
                if sd_num < k:
                    # if neighbors_distances[sd_idx] <= self.config.PREPROCESS.max_dist and [node_index, neighbors[sd_idx]] not in edges:
                    # print(node_index, sd_idx)
                    if [node_index, neighbors[sd_idx]] not in edges and [neighbors[sd_idx], node_index] not in edges:
                        edges.append([neighbors[sd_idx], node_index])
                        edges.append([node_index, neighbors[sd_idx]])
                else: break

        return [e[0] for e in edges], [e[1] for e in edges]

    def __fromIMG():
        #TODO
        return
    
    def __fromPDF():
        #TODO
        return

    def __fromNAF(self, src = NAF / 'simple/test'):

        #! CHANGE THIS IF WANTS TO UPLOAD DIFFERENT DATASET
        split_file = NAF / 'simple_train_valid_test_split.json'
        organize_naf(split_file, 'simple')

        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'images': [], 'paths': [], 'texts': [], 'boxs': []}
        
        justOne = random.choice(os.listdir(src)).split(".")[0]
        annotations = [file for file in os.listdir(src) if file.split(".")[1] == 'json']

        for file in tqdm(annotations, desc='Creating graphs:'):
            name = file.split('.')[0]
            img = Image.open(os.path.join(src, f'{name}.jpg')).convert('RGB')
            features['images'].append(img)
            features['paths'].append(os.path.join(src, f'{name}.jpg'))
            with open(os.path.join(src, file), 'r') as f:
                annotations = json.load(f)
            
            # getting infos
            boxs, texts, nl, ids = list(), list(), list(), list()
            pair_labels = dict()

            for key in ['fieldBBs', 'textBBs']:
                for e in annotations[key]:
                    ids.append(e['id'])
                    boxs.append(e['poly_points'][0] + e['poly_points'][2])
                    nl.append(key)
                    try:
                        texts.append(annotations["transcriptions"][e['id']])
                    except:
                        texts.append("")
            
            for key in ['pairs', 'samePairs']:
                for pair in annotations[key]:
                    edge = str(ids.index(pair[0])) + '-' + str(ids.index(pair[1]))
                    pair_labels[edge] = key
                    edge = str(ids.index(pair[1])) + '-' + str(ids.index(pair[0]))
                    pair_labels[edge] = key

            features['boxs'].append(boxs)
            features['texts'].append(texts)
            node_labels.append(nl)

            # getting edges
            node_ids = range(len(boxs))
            if self.edge_type == 'fully':
                u, v = self.__fully_connected(node_ids)
            elif self.edge_type == 'knn':
                u,v = self.__knn(img.size, boxs)
            else:
                raise Exception('Other edge types still under development.')
            
            el = list()
            for e in zip(u, v):
                edge = str(e[0]) + '-' + str(e[1])
                if edge in pair_labels.keys(): el.append(pair_labels[edge])
                else: el.append('none')
            edge_labels.append(el)

            # creating graph
            g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(boxs), idtype=torch.int32)
            graphs.append(g)

            if name == justOne:
                
                print("\n\n### EXAMPLE ###")
                img_path = os.path.join(src, f'{name}.jpg')
                print("Savin example:", img_path)

                center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

                # Saving original with labels
                img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(img)
                for box in boxs:
                    draw.rectangle(box, outline='blue', width=3)
                for key in ['pairs', 'samePairs']:
                    for pair in annotations[key]:
                        sc = center(boxs[ids.index(pair[0])])
                        ec = center(boxs[ids.index(pair[1])])
                        if key == 'pairs': color = 'orange'
                        else: color = 'green'
                        draw.line((sc,ec), fill=color, width=5)
                img.save(f'esempi/{name}_label.png')

                # Saving with graph
                num_none = 0
                num_pair = 0
                num_same = 0

                img_links = Image.open(img_path).convert('RGB')
                draw_links = ImageDraw.Draw(img_links)

                for box in boxs:
                    draw_links.rectangle(box, outline='blue', width=3)

                for p, pair in enumerate(el):
                    sc = center(boxs[u[p]])
                    ec = center(boxs[v[p]])
                    if pair == 'pairs': 
                        color = 'orange'
                        num_pair += 1
                    elif pair == 'samePairs': 
                        color = 'green'
                        num_same += 1
                    else: 
                        num_none += 1
                        color='gray'
                    draw_links.line((sc,ec), fill=color, width=5)
                
                print("Links: None {} | Key-Value {} | SameEntity {}".format(num_none, num_pair, num_same))
                img_links.save(f'esempi/{name}_knn_graph.png')

                # Print balanced edge graph
                edge_unique_labels = np.unique(el)
                g.edata['label'] = torch.tensor([np.where(target == edge_unique_labels)[0][0] for target in el])

                g = self.balance_edges(g, 3, 0)
                num_none = 0
                num_pair = 0
                num_same = 0

                img_removed = Image.open(os.path.join(src, f'{name}.jpg')).convert('RGB')
                draw_removed = ImageDraw.Draw(img_removed)
                for box in boxs:
                    draw_removed.rectangle(box, outline='blue', width=3)

                u, v = g.all_edges()
                labels = g.edata['label'].tolist()
                u, v = u.tolist(), v.tolist()

                for p, pair in enumerate(zip(u,v)):
                    sc = center(boxs[pair[0]])
                    ec = center(boxs[pair[1]])
                    if labels[p] == 1: 
                        num_pair += 1
                        color = 'orange'
                    elif labels[p] == 2: 
                        num_same += 1
                        color = 'green'
                    else: 
                        num_none += 1
                        color='gray'
                    draw_removed.line((sc,ec), fill=color, width=5)
                
                print("Balanced Links: None {} | Key-Value {} | SameEntity {}".format(num_none, num_pair, num_same))
                img_removed.save(f'esempi/{name}_removed_graph.png')
        return graphs, node_labels, edge_labels, features
    
    def __fromPAU(self, src: str):
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
                        # print(file_gt)
                        region_label = child[0].attrib['value']
                        region_bbox = [int(child[1].attrib['points'].split(" ")[0].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[1].split(",")[1].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[2].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[3].split(",")[1].split(".")[0])]
                        regions.append([region_label, region_bbox])
                        # print([region_label, region_bbox])

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
                                            #print([word_text, word_bbox])
            
            features['boxs'].append(tokens_bbox)
            features['texts'].append(tokens_text)
            node_labels.append(nl)

            # getting edges
            if self.edge_type == 'fully':
                u, v = self.__fully_connected(range(len(tokens_bbox)))
            elif self.edge_type == 'knn': 
                u,v = self.__knn(Image.open(os.path.join(src, image)).size, tokens_bbox)
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

    def __fromFUNSD(self, src : str):
        """ Parsing FUNSD annotation json files
        """
        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'paths': [], 'texts': [], 'boxs': []}
        # justOne = random.choice(os.listdir(os.path.join(src, 'adjusted_annotations'))).split(".")[0]
        test = src.split("/")[-1] == 'testing_data'
        if not test:
            for file in tqdm(os.listdir(os.path.join(src, 'adjusted_annotations')), desc='Creating graphs'):
            
                img_name = f'{file.split(".")[0]}.jpg'
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
                    # linking (list) and id (int)
                
                for p, pair in enumerate(pair_labels):
                    pair_labels[p] = [ids.index(pair[0]), ids.index(pair[1])]
                
                node_labels.append(nl)
                features['texts'].append(texts)
                features['boxs'].append(boxs)
                
                # getting edges
                if self.edge_type == 'fully':
                    u, v = self.__fully_connected(range(len(boxs)))
                elif self.edge_type == 'knn': 
                    u,v = self.__knn(Image.open(img_path).size, boxs)
                else:
                    raise Exception('Other edge types still under development.')
                
                el = list()
                for e in zip(u, v):
                    edge = [e[0], e[1]]
                    # reverse_edge = [e[1], e[0]]
                    if edge in pair_labels: el.append('pair')
                    # elif reverse_edge in pair_labels: el.append('pair')
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

        else:
            path_preds = '/home/gemelli/projects/doc2graph/src/data/test_bbox'
            path_images = '/home/gemelli/projects/doc2graph/DATA/FUNSD/testing_data/images'
            path_gts = '/home/gemelli/projects/doc2graph/DATA/FUNSD/testing_data/adjusted_annotations'
            all_preds, all_links, all_labels = load_predictions('test', path_preds, path_gts, path_images)
            print("FUNSD TESTING !!!!!!!!!!!")
            for f, file in enumerate(tqdm(os.listdir(os.path.join(src, 'adjusted_annotations')), desc='Creating graphs')):
            
                img_name = f'{file.split(".")[0]}.jpg'
                img_path = os.path.join(src, 'images', img_name)
                features['paths'].append(img_path)

                boxs, texts, nl = all_preds[f], list(), all_labels[f]
                pair_labels = all_links[f]

        return graphs, node_labels, edge_labels, features
