import os
from itsdangerous import json
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn
import torch.nn.functional as F
import dgl
from datetime import datetime
import shutil
import yaml

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
        self.early_stop = False
        self.model = model
        self.metric = metric
        e = datetime.now()
        if name == '': self.name = f'{e.strftime("%Y%m%d-%H%M")}'
        else: self.name = name

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()

        if self.metric == 'loss':
            if score > self.best_score:
                self.counter += 1
                print(f'    !- Stop Counter {self.counter} / {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print(f'    !- Validation LOSS decreased from {self.best_score} -> {score}')
                self.best_score = score
                self.save_checkpoint()
                self.counter = 0

        elif self.metric == 'acc':
            if score <= self.best_score:
                self.counter += 1
                print(f'    !- Stop Counter {self.counter} / {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print(f'    !- Validation ACCURACY increased from {self.best_score} -> {score}')
                self.best_score = score
                self.save_checkpoint()
                self.counter = 0
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

def get_binary_accuracy_and_f1(scores, labels, per_class = False):

    out = torch.nn.Sigmoid()
    scores = torch.where(out(scores) > 0.5, 1, 0)
    correct = torch.sum(scores == labels)
    accuracy = correct.item() * 1.0 / len(labels)
    classes = scores.detach().cpu().numpy()
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

def get_features(text, visual, edges):
    feat_n = 'bbox'
    feat_e = 'false'

    if text and visual:
        feat_n = 'all'
    elif text:
        feat_n = 'text'
    elif visual:
        feat_n = 'visual'
    
    if edges:
        feat_e = 'true'
        
    return feat_n, feat_e

def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores.to(device), labels.to(device))

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_auc_all(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    return roc_auc_score(labels, scores)