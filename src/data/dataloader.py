from random import randint
from typing import Tuple
import torch
import torch.utils.data as data
import os
import numpy as np
import dgl
from PIL import Image, ImageDraw

from src.data.feature_builder import FeatureBuilder
from src.data.graph_builder import GraphBuilder
from src.utils import get_config

class Document2Graph(data.Dataset):
    """This class convert documents (both images or pdfs) into graph structures.
    """

    def __init__(self, name : str, src_path : str, device = str, output_dir = str):
        """
        Args:
            name (str): should be one of the following: ['gt', 'img', 'pdf']
            src_path (str): path to folder containing documents
            device (str): device to use. can be 'cpu' or 'cuda:n'
            output_dir (str): where to save printed graphs examples
        """

        # initialize class
        if not os.path.isdir(src_path): raise Exception(f'src_path {src_path} does not exists\n -> please provide an existing path')

        self.name = name
        self.src_path = src_path
        self.cfg_preprocessing = get_config('preprocessing')
        self.src_data = self.cfg_preprocessing.LOADER.src_data
        self.GB = GraphBuilder()
        self.FB = FeatureBuilder(device)
        self.output_dir = output_dir

        # TODO: DO A DIFFERENT FILE
        self.COLORS = {'invoice_info': (150, 75, 0), 'receiver':(0,100,0), 'other':(128, 128, 128), 'supplier': (255, 0, 255), 'positions':(255,140,0), 'total':(0, 255, 255)}

        # get graphs
        self.graphs, self.node_labels, self.edge_labels, self.paths = self.__docs2graphs()
        
        # LABELS to numeric value
        # NODES
        if self.node_labels:
            self.node_unique_labels = np.unique(np.array([l for nl in self.node_labels for l in nl]))
            self.node_num_classes = len(self.node_unique_labels)
            self.node_num_features = self.graphs[0].ndata['feat'].shape[1]
        
            for idx, labels in enumerate(self.node_labels):
                self.graphs[idx].ndata['label'] = torch.tensor([np.where(target == self.node_unique_labels)[0][0] for target in labels], dtype=torch.int64)
        
        # EDGES
        if self.edge_labels:
            self.edge_unique_labels = np.unique(self.edge_labels[0])
            self.edge_num_classes = len(self.edge_unique_labels)
            try:
                # TODO to be changed
                self.edge_num_features = self.graphs[0].edata['feat'].shape[1]
            except:
                self.edge_num_features = 0
        
            for idx, labels in enumerate(self.edge_labels):
                self.graphs[idx].edata['label'] = torch.tensor([np.where(target == self.edge_unique_labels)[0][0] for target in labels], dtype=torch.int64)

    def __getitem__(self, index: int) -> dgl.DGLGraph:
        """ Returns item (graph), given index

        Args:
            index (int): index of the item to be taken.
        """
        return self.graphs[index]
    
    def __len__(self) -> int:
        """ Returns data length
        """
        return len(self.graphs)
    
    def __docs2graphs(self) -> Tuple[list, list, list, list]:
        """It uses GraphBuilder and FeaturesBuilder objects to get graphs (and lables, if any) from source data.

        Returns:
            tuple (lists): DGLGraphs, nodes and edges label names, paths per each file
        """
        graphs, node_labels, edge_labels, features = self.GB.get_graph(self.src_path, self.src_data)
        self.feature_chunks, self.num_mods = self.FB.add_features(graphs, features)
        return graphs, node_labels, edge_labels, features['paths']
    
    def label2class(self, label : str, node=True) -> int:
        """ Transform a label (str) into its class number.

        Args:
            label (str): node or edge string label
            node (bool): either if taking node (true) ord edge (false) class number
        
        Returns:
            int: class number
        """
        if node:
            return self.node_unique_labels[label]
        else:
            return self.edge_unique_labels[label]
    
    def get_info(self, num_graph=0) -> None:
        """ Print information regarding the data uploaded

        Args:
            num_graph (int): give one index to print one example graph information
        """
        print(f"\n{self.name.upper()} dataset:\n-> graphs: {len(self.graphs)}\n-> node labels: {self.node_unique_labels}\n-> edge labels: {self.edge_unique_labels}\n-> node features: {self.node_num_features}")
        self.GB.get_info()
        self.FB.get_info()
        print(f"-> graph example: {self.graphs[num_graph]}")
        return
    
    def balance(self, cls = 'none', indices = None) -> None:
        """ Calls balance_edges() of GraphBuilder.

        Args:
            cls (str): 
        """

        cls = int(np.where(cls == self.edge_unique_labels)[0][0])
        if indices is None:
            for i, g in enumerate(self.graphs):
                self.graphs[i] = self.GB.balance_edges(g, self.edge_num_classes, cls = cls)
        else:
            for id in indices:
                self.graphs[id] = self.GB.balance_edges(self.graphs[id], self.edge_num_classes, cls = cls)
        
        return
    
    def get_chunks(self) -> list:
        """ get feature_chunks, meaning the length of different modalities (features) contributions inside nodes.

        Returns:
            feature_chunks (list) : list of feature chunks
        """
        if len(self.feature_chunks) != self.num_mods: self.feature_chunks.pop(0)
        return self.feature_chunks
    
    def print_graph(self, num=None, node_labels=None, labels_ids=None, name='doc_graph', bidirect=True, regions=[], preds=None) -> Image:
        """ Print a given graph over its image document.

        Args:
            num (int): which graph / document to print
            node_labels (list): list of node labels
            labels_ids (list): list of labels ids to print
            name (str): name to give to output file
            bidirect (bool): either to print the graph bidirected or not
            regions (list): debug purposes for layout anaylis, if any available it prints it
            preds (list): if any, it prints model predictions
        
        Returns:
            graph_img (Image) : the drawn graph over the document
        """
        if num is None: num = randint(0, self.__len__()-1)
        graph = self.graphs[num]
        graph_path = self.paths[num]
        graph_img = Image.open(graph_path).convert('RGB')
        if labels_ids is None: labels_ids = graph.edata['label'].nonzero().flatten().tolist()
        center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
        graph_draw = ImageDraw.Draw(graph_img)
        w, h = graph_img.size
        boxs = graph.ndata['geom'][:, :4].tolist()
        boxs = [[box[0]*w, box[1]*h, box[2]*w, box[3]*h] for box in boxs]

        if node_labels is not None:
            for b, box in enumerate(boxs):
                label = self.node_unique_labels[node_labels[b]]
                graph_draw.rectangle(box, outline=self.COLORS[label], width=2)
        else:
            for box in boxs:
                graph_draw.rectangle(box, outline='blue', width=2)
        
        for region in regions:
            color = self.COLORS[region[0]]
            graph_draw.rectangle(region[1], outline=color, width=4)
        
        if preds is not None:
            graph_draw.rectangle(preds, outline='green', width=4)
        
        u,v = graph.edges()
        for id in labels_ids:
            sc = center(boxs[u[id]])
            ec = center(boxs[v[id]])
            graph_draw.line((sc,ec), fill='violet', width=3)
            if bidirect:
                graph_draw.ellipse([(sc[0]-4,sc[1]-4), (sc[0]+4,sc[1]+4)], fill = 'green', outline='black')
                graph_draw.ellipse([(ec[0]-4,ec[1]-4), (ec[0]+4,ec[1]+4)], fill = 'red', outline='black')

        graph_img.save(self.output_dir / f'{name}.png')
        return graph_img
        