from cProfile import label
import os
from pickletools import optimize
from itsdangerous import json
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import torch
import torch.nn
import torch.nn.functional as F
import dgl
from datetime import datetime
import shutil
import yaml
import numpy as np

from src.paths import CONFIGS, OUTPUTS, RESULTS, WEIGHTS


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

    def step(self, score):
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

    def save_checkpoint(self):
        '''Saves model when validation acc increase.'''
        torch.save(self.model.state_dict(), WEIGHTS / f'{self.name}.pt')

def save_best_results(best_params : dict, rm_logs : bool = False) -> None:
    """Save best results for cross validation.

    Args:
        best_params (dict): best parameters among k-fold cross validation.
        rm_logs (bool, optional): Remove tmp weights in output folder if True. Defaults to False.
    """
    models = OUTPUTS / 'tmp'
    output = WEIGHTS / best_params['model']
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
        filename (str): _description_
        infos (_type_): _description_
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

def get_binary_accuracy_and_f1(classes, labels, per_class = False):

    print(classes.shape, labels.shape)
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

def validate(model, graphs : list, device : str, loss_fcn) -> float:
    """_summary_

    Args:
        model (_type_): _description_
        graphs (list): _description_
        device (str): _description_
        loss_fcn (_type_): _description_

    Returns:
        float: _description_
    """
    model.eval()
    val_acc = 0
    with torch.no_grad():
        all_graphs = dgl.batch(graphs).to(device)
        feat = all_graphs.ndata['feat'].to(device)
        target = all_graphs.ndata['label'].to(device)
        logits, attn = model(all_graphs, feat)
        test_acc = accuracy(logits, target)
        loss = loss_fcn(logits, target)
    return test_acc, loss.item()

def evaluate(model, graphs, device):
    model.eval()
    test_acc = 0
    with torch.no_grad():
        all_graphs = dgl.batch(graphs).to(device)
        feat = all_graphs.ndata['feat'].to(device)
        target = all_graphs.ndata['label'].to(device)
        logits, attn = model(all_graphs, feat)
        test_acc = accuracy(logits, target)
        macro, micro = get_f1(logits, target)
    return test_acc, (macro, micro)

def get_device(value : int) -> str:
    """Either to use cpu or gpu (and which one).
    """
    if value < 0: 
        return 'cpu'
    else: 
        return 'cuda:{}'.format(value)

def get_features(args):
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

def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    s = F.sigmoid(scores)
    print("\n", scores.shape, torch.max(s).item(), s.min().item(), ((s < 0.5).sum()/s.shape[0]).item(), pos_score.shape, neg_score.shape, "\n")
    input()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    w = torch.ones_like(labels)
    w[pos_score.shape[0]:] = 0.01
    return F.binary_cross_entropy_with_logits(scores.flatten().to(device), labels.to(device), weight=w.to('cuda:0'))

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_loss_2(scores : torch.Tensor, labels : torch.Tensor):
    
    s = torch.sigmoid(scores)
    # print("\n", scores.shape, torch.max(s).item(), s.min().item(), ((s < 0.5).sum()/s.shape[0]).item(), "\n")
    # w = torch.ones_like(labels)
    # w[pos_score.shape[0]:] = 0.01
    # print(scores.dtype, labels.dtype)
    w = class_weight.compute_sample_weight(class_weight='balanced', y=labels.cpu().numpy())
    # w = [1e6 if (l == 1) else 0.1 for l in labels]
    # print(np.unique(w))
    return F.binary_cross_entropy_with_logits(scores, labels, weight=torch.tensor(w).to('cuda:0'))
    # return F.binary_cross_entropy_with_logits(scores, labels)

def compute_crossentropy_loss(scores : torch.Tensor, labels : torch.Tensor):
    w = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to('cuda:0'))(scores, labels)

def compute_auc_2(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    # return roc_auc_score(labels, scores)
    return average_precision_score(labels, scores)

def compute_auc_mc(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = F.one_hot(labels).cpu().numpy()
    # return roc_auc_score(labels, scores)
    return average_precision_score(labels, scores)

def compute_auc_all(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    optimal_thres = find_optimal_cutoff(labels, scores)
    return roc_auc_score(labels, scores), optimal_thres

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
