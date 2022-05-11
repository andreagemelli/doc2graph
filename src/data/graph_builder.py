from cgitb import text
import json
import os
from PIL import Image
from src.utils import get_config
import torch
import dgl


class GraphBuilder():

    def __init__(self) -> None:
        self.cfg_preprocessing = get_config('preprocessing')
        self.edge_type = self.cfg_preprocessing.GRAPHS.edge_type
        self.data_type = self.cfg_preprocessing.GRAPHS.data_type
        return
    
    def get_graph(self, src_path, src_data):
        
        if src_data == 'FUNSD':
            return self.__fromFUNSD(src_path)
        elif src_data == 'NAF':
            return self.__fromNAF()
        elif src_data == 'CUSTOM':
            if self.data_type == 'img':
                return self.__fromIMG()
            elif self.data_type == 'pdf':
                return self.__fromPDF()
            else:
                raise Exception('GraphBuilder exception: data type invalid. Choose from ["img", "pdf"]')
        else:
            raise Exception('GraphBuilder exception: source data invalid. Choose from ["FUNSD", "NAF", "CUSTOM"]')
    
    def get_info(self):
        print(f"-> edge_type: {self.edge_type[0]}")

    def __fully_connected(self, ids : list):
        u, v = list(), list()
        for id in ids:
            u.extend([id for i in range(len(ids)) if i != id])
            v.extend([i for i in range(len(ids)) if i != id])
        return torch.tensor(u), torch.tensor(v)

    def __fromIMG():
        #TODO
        return
    
    def __fromPDF():
        #TODO
        return

    def __fromNAF():
        #TODO
        #! clue: use 'isBlank' info as features (categorical encoding)
        #! import also 'types' as node labels (if we can, we perform also entity recognition)
        return

    def __fromFUNSD(self, src : str):
        """ Parsing FUNSD annotation json files
        """
        graphs, labels = list(), list()
        features = {'images': [], 'texts': [], 'boxs': []}
        for file in os.listdir(os.path.join(src, 'annotations')):
            
            img = Image.open(os.path.join(src, 'images', f'{file.split(".")[0]}.png')).convert('RGB')
            features['images'].append(img)

            with open(os.path.join(src, 'annotations', file), 'r') as f:
                form = json.load(f)['form']

            # getting infos
            boxs, texts, g_labels = list(), list(), list()

            for elem in form:
                if elem['text']:
                    boxs.append(elem['box'])
                    texts.append(elem['text'])
                    g_labels.append(elem['label'])
            
            features['texts'].append(texts)
            features['boxs'].append(boxs)
            
            # getting edges
            node_ids = range(len(boxs))
            if self.edge_type == 'fully':
                u, v = self.__fully_connected(node_ids)
            else:
                raise Exception('Other edge types still under development.')

            # creating graph
            g = dgl.graph((u, v), num_nodes=len(boxs), idtype=torch.int32)
            graphs.append(g), labels.append(g_labels)

        return graphs, labels, features
