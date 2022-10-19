from argparse import ArgumentParser
from cProfile import label
import os
from pickletools import optimize
from typing import Tuple
from itsdangerous import json
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import torch
import torch.nn
import torch.nn.functional as F
import dgl
from datetime import datetime
import shutil
import yaml
import numpy as np

from src.paths import CHECKPOINTS, CONFIGS, OUTPUTS, RESULTS


class EarlyStopping:
    """Early stop for training purposes, looking at validation loss.
    """
    def __init__(self, model, name = '', metric = 'loss', patience=50):
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
        e = datetime.now()
        if name == '': self.name = f'{e.strftime("%Y%m%d-%H%M")}'
        else: self.name = name

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
        torch.save(self.model.state_dict(), CHECKPOINTS / f'{self.name}.pt')

def save_best_results(best_params : dict, rm_logs : bool = False) -> None:
    """Save best results for cross validation.

    Args:
        best_params (dict): best parameters among k-fold cross validation.
        rm_logs (bool, optional): Remove tmp weights in output folder if True. Defaults to False.
    """
    models = OUTPUTS / 'tmp'
    output = CHECKPOINTS / best_params['model']
    shutil.copyfile(models / best_params['model'], output)

    new_configs = CONFIGS / (best_params['model'].split(".")[0] + '.yaml')
    shutil.copyfile(CONFIGS / 'base.yaml', new_configs)

    with open(new_configs) as f:
        config = yaml.safe_load(f)

    config['MODEL']['num_layers'] = best_params['num_layers']
    config['TRAIN']['batch_size'] = best_params['batch_size']
    config['INFO'] = {'split': best_params['split'], 'val_loss': best_params['val_loss'], 'total_params': best_params['total_params']}

    with open(new_configs, 'w') as f:
        yaml.dump(config, f)

    if rm_logs and os.path.isdir(models):
        shutil.rmtree(models)

    return

def save_test_results(filename : str, infos : dict) -> None:
    """Save test results.

    Args:
        filename (str): name of the file to save results of experiments
        infos (dict): what to save in the json file about training
    """
    results = RESULTS / (filename + '.json')

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

def accuracy(logits : torch.Tensor, labels : torch.Tensor) -> float:
    """Accuracy of the model.

    Args:
        logits (torch.Tensor): model prediction logits
        labels (torch.Tensor): target labels

    Returns:
        float: accuracy
    """
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def get_device(value : int) -> str:
    """Either to use cpu or gpu (and which one).
    """
    if value < 0: 
        return 'cpu'
    else: 
        return 'cuda:{}'.format(value)

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

def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def generalized_f1_score(y_true, y_pred, match):
    # y_true = (y_nodes, y_link)
    # y_pred = (y_nodes, y_link)

    # # nodes
    # micro_f1_nodes, macro_f1_nodes = 0, 0
    
    nodes_confusion_mtx = confusion_matrix(y_true=y_true[0][list(match["pred2gt"].keys())], y_pred=y_pred[0][list(match["gt2pred"].values())], 
                                           labels=[0, 1, 2, 3], normalize=None)
    print(nodes_confusion_mtx)
    ntp = [nodes_confusion_mtx[i, i] for i in range(nodes_confusion_mtx.shape[0])]
    nfp = [(nodes_confusion_mtx[:i, i].sum() + nodes_confusion_mtx[i+1:, i].sum()) for i in range(nodes_confusion_mtx.shape[0])]
    nfn = [(nodes_confusion_mtx[i, :i].sum() + nodes_confusion_mtx[i, i+1:].sum()) for i in range(nodes_confusion_mtx.shape[0])]

    macro_f1_nodes = np.mean([tp / (tp + (0.5 * (fp + len(match["false_positive"]) + fn + len(match["false_negative"])))) for (tp, fp, fn) in zip(ntp, nfp, nfn)])
    micro_f1_nodes = np.sum(ntp) / (np.sum(ntp) + (0.5 * (np.sum(nfp) + len(match["false_positive"]) + np.sum(nfn) + len(match["false_negative"]))))

    # links
    # micro_f1_links, macro_f1_links = 0, 0

    # links2keep = [idx for idx, link in enumerate(y_true[1]) if (link[0] in match["pred2gt"].values()) and (link[1] in match["pred2gt"].values())]
    links_confusion_mtx = confusion_matrix(y_true=y_true[1], y_pred=y_pred[1], labels=[0, 1], normalize=None)

    ltp = [links_confusion_mtx[i, i] for i in range(links_confusion_mtx.shape[0])]
    lfp = [(links_confusion_mtx[:i, i].sum() + links_confusion_mtx[i+1:, i].sum()) for i in range(links_confusion_mtx.shape[0])]
    lfn = [(links_confusion_mtx[i, :i].sum() + links_confusion_mtx[i, i+1:].sum()) for i in range(links_confusion_mtx.shape[0])]

    # n = len(links2keep) + len(match["false_positive"]) - 1
    macro_f1_links = np.mean([tp / (tp + (0.5 * (fp + fn + match["n_link_fn"]))) for (tp, fp, fn) in zip(ltp, lfp, lfn)])
    # if match['n_link_fn'] == 0: f1_pairs = None
    f1_pairs = [tp / (tp + (0.5 * (fp + fn + match["n_link_fn"])) + 1e-6) for (tp, fp, fn) in zip(ltp, lfp, lfn)][1]
    micro_f1_links = np.sum(ltp) / (np.sum(ltp) + (0.5 * (np.sum(lfp) + np.sum(lfn) + match["n_link_fn"])) + 1e-6)

    return ntp, nfp, nfn, ltp, lfp, lfn, {"nodes": {"micro_f1": micro_f1_nodes, "macro_f1": macro_f1_nodes}, "links": {"micro_f1": micro_f1_links, "macro_f1": macro_f1_links}, "pairs_f1": f1_pairs}
