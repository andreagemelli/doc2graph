import json
import os
from PIL import Image
from src.data.amazing_utils import organize_naf
from src.paths import NAF
from src.amazing_utils import get_config
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
    
    def get_info(self):
        print(f"-> edge_type: {self.edge_type[0]}")

    def __fully_connected(self, ids : list):
        u, v = list(), list()
        for id in ids:
            u.extend([id for i in range(len(ids)) if i != id])
            v.extend([i for i in range(len(ids)) if i != id])
        return u, v

    def __fromIMG():
        #TODO
        return
    
    def __fromPDF():
        #TODO
        return

    def __fromNAF(self, src = NAF / 'simple/test'):
        #TODO
        #! import also 'types' as node labels (if we can, we perform also entity recognition)
        # FUNSD
        # model = FUDGE / 'saved/FUNSDLines_detect_augR_staggerLighter.pth'
        # TRAINED_MODEL = FUDGE / "saved/FUNSDLines_pair_graph663rv_new/checkpoint-iteration700000.pth"

        # NAF
        # sys.path.append(os.path.join(ROOT,'FUDGE'))
        # from FUDGE.run import detect_boxes
        # model = FUDGE / 'saved/NAF_detect_augR_staggerLighter.pth'
        # img = transform_image(os.path.join(NAF, "groups/123/100572410_00029.jpg"))

        # boxes, visual_features = detect_boxes(
        #     img,
        #     img_path = os.path.join(NAF, "groups/123/100572410_00029.jpg"),
        #     output_path=ROOT,
        #     include_threshold= 0.5,
        #     model_checkpoint = model,
        #     device='cuda:0',
        #     debug=True)

        #! CHANGE THIS IF WANTS TO UPLOAD DIFFERENT DATASET
        split_file = NAF / 'simple_train_valid_test_split.json'
        organize_naf(split_file, 'simple')

        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'images': [], 'texts': [], 'boxs': []}

        for file in os.listdir(src):
            name, extension = file.split('.')[0], file.split('.')[1]
            if extension == 'json':
                img = Image.open(os.path.join(src, f'{name}.jpg')).convert('RGB')
                features['images'].append(img)
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

                features['boxs'].append(boxs)
                features['texts'].append(texts)
                node_labels.append(nl)

                # getting edges
                node_ids = range(len(boxs))
                if self.edge_type == 'fully':
                    u, v = self.__fully_connected(node_ids)
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

        return graphs, node_labels, edge_labels, features

    def __fromFUNSD(self, src : str):
        """ Parsing FUNSD annotation json files
        """
        graphs, node_labels = list(), list()
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
            g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(boxs), idtype=torch.int32)
            graphs.append(g), node_labels.append(g_labels)

        return graphs, node_labels, None, features
