import torch
from src.amazing_utils import create_folder
from src.training.amazing_utils import EarlyStopping, accuracy, get_device, get_f1, validate
from src.data.dataloader import Document2Graph
from src.paths import NAF, OUTPUTS
from src.training.models import EdgeClassifier
import dgl
from sklearn.utils import class_weight
import numpy as np


def link_and_rec(args):

    device = get_device(args.gpu)

    train_data = Document2Graph('TRAIN', NAF / 'simple/train', device)
    #train_data.balance()
    train_data.get_info()

    # train_data = Document2Graph('VALID', NAF / 'simple/valid', device)
    # # train_data.balance()
    # train_data.get_info()

    valid_data = Document2Graph('VALID', NAF / 'simple/valid', device)
    # valid_data.balance()
    valid_data.get_info()

    graphs = dgl.batch(train_data.graphs).to(device)
    node_features = graphs.ndata['feat'].to(device)
    edge_targets = graphs.edata['label'].to(device)

    # class_weights=class_weight.compute_class_weight(class_weight='balanced', 
    #                                                 classes=np.unique(edge_targets.cpu().detach().numpy()), 
    #                                                 y=edge_targets.cpu().detach().numpy())
    
    # print(class_weights)
    # TODO change input number of features, to be evaluated innerly
    model = EdgeClassifier(200, 1000, train_data.edge_num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    #loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    loss_fcn = torch.nn.CrossEntropyLoss()

    val_graphs = dgl.batch(valid_data.graphs).to(device)
    val_node_features = val_graphs.ndata['feat'].to(device)
    val_edge_targets = val_graphs.edata['label'].to(device)

    num_epochs = 2000
    stopper = EarlyStopping(model, name='edges', metric='acc', patience=200)

    for epoch in range(num_epochs):
        model.train()
        logits = model(graphs, node_features)
        train_loss = loss_fcn(logits, edge_targets)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        train_acc = accuracy(logits, edge_targets)
        train_f1, _ = get_f1(logits, edge_targets)

        model.eval()
        with torch.no_grad():
            val_preds = model(val_graphs, val_node_features)
            val_acc = accuracy(val_preds, val_edge_targets)
            val_loss = loss_fcn(val_preds, val_edge_targets)
            val_f1, _ = get_f1(val_preds, val_edge_targets)
        
        print("\nEpoch {}/{}:\n - Accuracy: Train {} | Validation {}\n - Loss: Train {} | Validation {}\n - F1 Macro: Train {} | Validation {}"
        .format(epoch+1, num_epochs, train_acc, val_acc, train_loss.item(), val_loss.item(), train_f1, val_f1))

        if stopper.step(val_f1): break
    
    test_data = Document2Graph('TEST', NAF / 'simple/test', device)
    test_data.get_info()

    test_graphs = dgl.batch(test_data.graphs).to(device)
    test_node_features = test_graphs.ndata['feat'].to(device)
    test_edge_targets = test_graphs.edata['label'].to(device)
    model.eval()
    test_acc = 0
    with torch.no_grad():
        logits= model(test_graphs, test_node_features)
        test_acc = accuracy(logits, test_edge_targets)
        macro, micro = get_f1(logits, test_edge_targets)
        classes_f1 = get_f1(logits, test_edge_targets, per_class=True)
    print("\nTEST RESULTS: Accuracy ", test_acc, " - F1 Marco ", macro)
    print("\nPrecision - Recall - F1")
    print("None:", classes_f1[0], "Pairs:", classes_f1[1], "SamePairs:", classes_f1[2])

    #! test_data.GB.balance_edges(test_data.graphs[test_graph], test_data.edge_num_classes, 0)
    create_folder(OUTPUTS / 'link_predictions')
    for test_graph in range(len(test_data.graphs)):
        ne = test_data.graphs[test_graph].num_edges()
        pne = sum([test_data.graphs[n].num_edges() for n in range(test_graph)])
        _, indices = torch.max(logits, dim=1)
        test_graph_indices = indices[pne:pne+ne]
        img = test_data.images[test_graph]
        size = img.size
        scale_back = lambda rect, s : [int(rect[0]*s[0]), int(rect[1]*s[1]), int(rect[2]*s[0]), int(rect[3]*s[1])]
        boxs = [scale_back(feat[:4].tolist(), size) for feat in test_data.graphs[test_graph].ndata['feat']]
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for b in boxs:
            draw.rectangle(b, outline='blue', width=3)
        
        center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
        start_nodes, end_nodes = test_data.graphs[test_graph].edges()
        for e, pred in enumerate(test_graph_indices):
            pred = int(pred)
            if pred == 0: color = 'grey'
            elif pred == 1: color = 'orange'
            elif pred == 2: color = 'green'
            start, end = center(boxs[start_nodes[e]]), center(boxs[end_nodes[e]])
            draw.line((start,end), fill=color, width=3)

        img.save(OUTPUTS / f'link_predictions/{test_graph}.png')

    return

def train_naf(args):

    if args.task == 'elab' or args.task == 'elin':
        link_and_rec(args)

    elif args.task == 'wgrp':
        raise Exception("Word Grouping not defined for NAF.")

    else:
        raise Exception("Task selected does not exists. Enter:\
            - 'elab': entity labeling\
            - 'elin': entity linking\
            - 'wgrp': word grouping")

    return