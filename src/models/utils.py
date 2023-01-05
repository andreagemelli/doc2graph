from argparse import ArgumentParser
from typing import Tuple
from itsdangerous import json
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.utils import class_weight
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import os

from src.paths import CHECKPOINTS, RESULTS
from src.utils import create_folder


class EarlyStopping:
    """Early stop for training purposes, looking at validation loss.
    """
    def __init__(self, model, name, metric = 'loss', patience=50):
        """Constructor.

        Args:
            model (DGLModel): graph or batch of graphs
            name (str): name for weights.
            metric (str): set the stopper, following loss ['loss'] or accuracy ['acc'] on validation
            patience (int, optional): if validation do not improve after 'patience' iters, it stops training. Defaults to 50.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = 'improved'
        self.model = model
        self.metric = metric
        self.name = name

    def step(self, score : float) -> str:
        """ It does a step of the stopper. If metric does not encrease after a while, it stops the training.

        Args:
            score (float) : metric / value to keep track of.

        Returns
            A status used by traning to do things.
        """

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
            return 'improved'

        if self.metric == 'loss':
            if score > self.best_score:
                self.counter += 1
                print(f'    !- Stop Counter {self.counter} / {self.patience}')
                self.early_stop = 'not-improved'
                if self.counter >= self.patience:
                    self.early_stop = 'stop'
            else:
                print(f'    !- Validation LOSS decreased from {self.best_score} -> {score}')
                self.best_score = score
                self.save_checkpoint()
                self.counter = 0
                self.early_stop = 'improved'

        elif self.metric == 'acc':
            if score <= self.best_score:
                self.counter += 1
                print(f'    !- Stop Counter {self.counter} / {self.patience}')
                self.early_stop = 'not-improved'
                if self.counter >= self.patience:
                    self.early_stop = 'stop'
            else:
                print(f'    !- Validation ACCURACY increased from {self.best_score} -> {score}')
                self.best_score = score
                self.save_checkpoint()
                self.counter = 0
                self.early_stop = 'improved'
                
        else:
            raise Exception('EarlyStopping Error: metric provided not valid. Select between loss or acc')

        return self.early_stop

    def save_checkpoint(self) -> None:
        '''Saves model when validation acc increase.'''
        self.model.save(CHECKPOINTS / self.name)

def save_test_results(filename : str, infos : dict) -> None:
    """Save test results.

    Args:
        filename (str): name of the file to save results of experiments
        infos (dict): what to save in the json file about training
    """
    results = CHECKPOINTS / filename.split('/')[0] / 'results.json'

    with open(results, 'w') as f:
        json.dump(infos, f)
    return

def get_f1(logits : torch.Tensor, labels : torch.Tensor, per_class = False) -> tuple:
    """Returns Macro and Micro F1-score for given logits / labels.

    Args:
        logits (torch.Tensor): model prediction logits
        labels (torch.Tensor): target labels

    Returns:
        tuple: macro-f1 and micro-f1
    """
    _, indices = torch.max(logits, dim=1)
    indices = indices.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    if not per_class:
        return f1_score(labels, indices, average='macro'), f1_score(labels, indices, average='micro')
    else:
        return precision_recall_fscore_support(labels, indices, average=None)[2].tolist()

def get_binary_accuracy_and_f1(classes, labels : torch.Tensor, per_class = False) -> Tuple[float, list]:

    correct = torch.sum(classes.flatten() == labels)
    accuracy = correct.item() * 1.0 / len(labels)
    classes = classes.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    if not per_class:
        f1 = f1_score(labels, classes, average='macro'), f1_score(labels, classes, average='micro')
    else:
        f1 = precision_recall_fscore_support(labels, classes, average=None)[2].tolist()
    
    return accuracy, f1

def get_features(args : ArgumentParser) -> Tuple[str, str]:
    """ Return description of the features used in the experiment

    Args:
        args (ArgumentParser) : your ArgumentParser
    """
    feat_n = ''
    feat_e = 'false'

    if args.add_geom:
        feat_n += 'geom-'
    if args.add_embs:
        feat_n += 'text-'
    if args.add_visual:
        feat_n += 'visual-'
    if args.add_hist:
        feat_n += 'histogram-'
    if args.add_eweights:
        feat_e = 'true'
        
    return feat_n, feat_e

def compute_crossentropy_loss(scores : torch.Tensor, labels : torch.Tensor):
    w = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to('cuda:0'))(scores, labels)

def compute_auc_mc(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = F.one_hot(labels).cpu().numpy()
    # return roc_auc_score(labels, scores)
    return average_precision_score(labels, scores)

def create_run_folder():
    higher_id = -1
    for f in os.listdir(CHECKPOINTS):
        if os.path.isdir(CHECKPOINTS / f):
            id = f.split('run')[1]
            if int(id) > higher_id:
                higher_id = int(id)
    higher_id += 1
    create_folder(CHECKPOINTS / f'run{higher_id}')
    return f'run{higher_id}'