from src.data.feature_builder import FeatureBuilder
from src.data.graph_builder import GraphBuilder
from src.utils import get_config
import torch
import torch.utils.data as data
import os
import numpy as np

class Document2Graph(data.Dataset):
    """This class convert documents (both images or pdfs) into graph structures.
    """

    def __init__(self, name : str, src_path : str, device = int) -> None:
        """
        Args:
            src_type (str): should be one of the following: ['gt', 'img', 'pdf']
            src_path (str): path to folder containing documents
            add_embs (bool): If True, use Spacy to convert text contents to word embeddings
        """

        # initialize class
        self.name = name
        self.src_path = src_path
        if not os.path.isdir(self.src_path): raise Exception(f'src_path {src_path} does not exists\n -> please provide an existing path')
        self.cfg_preprocessing = get_config('preprocessing')
        self.src_data = self.cfg_preprocessing.LOADER.src_data
        self.gb = GraphBuilder()
        self.fb = FeatureBuilder(device)

        # get graphs
        self.graphs, self.labels = self.docs2graphs()

        # Labels to numeric value
        if self.labels:
            self.unique_labels = np.unique(self.labels[0])
            self.num_classes = len(self.unique_labels)
            self.num_features = self.graphs[0].ndata['feat'].shape[1]
        
            for idx, g_labels in enumerate(self.labels):
                self.graphs[idx].ndata['label'] = torch.tensor([np.where(target == self.unique_labels)[0][0] for target in g_labels])
    
    def __getitem__(self, index):
        # Return indexed graph
        return self.graphs[index]
    
    def __len__(self):
        # Return dataset length
        return len(self.graphs)
    
    def docs2graphs(self):
        """It uses GraphBuilder and FeaturesBuilder objects to get graphs (and lables, if any) from source data.

        Returns:
            tuple: DGL Graph and label
        """
        graphs, labels, features = self.gb.get_graph(self.src_path, self.src_data)
        self.fb.add_features(graphs, features)
        return graphs, labels
    
    def label2class(self, label):
        # Converts the numeric label to the corresponding string
        return self.unique_labels[label]
    
    def get_info(self):
        print(f"\n{self.name.upper()} dataset:\n-> graphs: {len(self.graphs)}\n-> labels: {self.unique_labels}\n-> num features: {self.num_features}")
        self.gb.get_info()
        self.fb.get_info()
        print(f"-> graph example: {self.graphs[0]}")