import torch
import torch.utils.data as data
import os
import numpy as np

from src.utils.dataloader import fromFUNSD

class DocumentGraphs(data.Dataset):
    """This class convert documents (both images or pdfs) into graph structures.
    """

    def __init__(self, name : str, src_path : str, src_type : str = 'file', add_embs : bool = False) -> None:
        """
        Args:
            src_type (str): should be one of the following: ['file', 'img', 'pdf']
            src_path (str): path to folder containing documents
            add_embs (bool): If True, use Spacy to convert text contents to word embeddings
        """

        # initialize class
        self.name = name
        self.src_path = src_path
        if not os.path.isdir(self.src_path): raise Exception(f'src_path {src_path} does not exists\n -> please provide an existing path')
        self.src_type = src_type
        if src_type not in ['file', 'img', 'pdf']: raise Exception(f"src_type {self.src_type} invalid\n -> should be one of the following ['file', 'img', 'pdf']")
        self.add_embs = add_embs
        
        # get graphs
        self.graphs, self.labels = self.docsToGraphs(self.src_type, self.src_path, self.add_embs)

        # Labels to numeric value
        self.unique_labels = np.unique(self.labels[0])
        self.num_classes = len(self.unique_labels)
        self.num_features = self.graphs[0].ndata['feat'].shape[1]
        
        for idx, g_labels in enumerate(self.labels):
            self.graphs[idx].ndata['label'] = torch.tensor([np.where(target == self.unique_labels)[0][0] for target in g_labels])
    
    def __getitem__(self, index):
        # Read the graph and label
        return self.graphs[index]
    
    def docsToGraphs(self, type : str, src : str, add_embs : bool):
        if type == 'file':
            #! call / write your custom dataset function
            return fromFUNSD(src, add_embs)

        elif type == 'img':
            #TODO Use OCR tool (e.g. Tesseract) to extract node bboxs and text contents
            raise Exception('img src_type not yet implemented\n -> select file as src_type')
        
        elif type == 'pdf':
            #TODO Use PyMuPDF tool to extract node bboxs and text contents
            raise Exception('pdf src_type not yet implemented\n -> select file as src_type')
    
    def label2class(self, label):
        # Converts the numeric label to the corresponding string
        return self.unique_labels[label]
    
    def get_info(self):
        print(f"\n{self.name.upper()} dataset:\n\
    -> graphs: {len(self.graphs)}\n\
    -> labels: {self.unique_labels}\n\
    -> num features: {self.num_features}\n\
    -> textual features: {self.add_embs}\n\
    -> graph example: {self.graphs[0]}")