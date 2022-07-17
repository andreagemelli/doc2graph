from random import seed
import torch
import dgl
import torch.nn.functional as F

from src.amazing_utils import create_folder, get_config
from src.training.amazing_utils import EarlyStopping, compute_auc_mc, compute_crossentropy_loss, get_binary_accuracy_and_f1, get_device, get_f1
from src.data.dataloader import Document2Graph
from src.paths import CHECKPOINTS, NAF, OUTPUTS
from src.models.graphs import SetModel


def link_and_rec(args):

    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:

        train_data = Document2Graph('TRAIN', NAF / 'simple/train', device)
        train_data.get_info()

        valid_data = Document2Graph('VALID', NAF / 'simple/valid', device)
        valid_data.get_info()

        graphs = dgl.batch(train_data.graphs).to(device)
        node_features = graphs.ndata['feat'].to(device)
        
        model = sm.get_model(None, 3, train_data.get_chunks())
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))

        val_graphs = dgl.batch(valid_data.graphs).to(device)
        val_node_features = val_graphs.ndata['feat'].to(device)

        stopper = EarlyStopping(model, name='naf', metric='acc', patience=1000)

        for epoch in range(cfg_train.epochs):
            model.train()
            logits = model(graphs, node_features)
            train_loss = compute_crossentropy_loss(logits.to(device), graphs.edata['label'].to(device))
            train_auc = compute_auc_mc(logits.to(device), graphs.edata['label'].to(device))
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(val_graphs, val_node_features)
                val_loss = compute_crossentropy_loss(val_preds.to(device), val_graphs.edata['label'].to(device))
                val_auc = compute_auc_mc(val_preds.to(device), val_graphs.edata['label'].to(device))
            
            print("\nEpoch {}/{}:\n - Loss: Train {} | Validation {}\n - AUC-PR: Train {} | Validation {}"
            .format(epoch+1, cfg_train.epochs, train_loss.item(), val_loss.item(), train_auc, val_auc))

            if stopper.step(val_auc) == 'stop': break
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        data = Document2Graph('VALID', NAF / 'simple/valid', device)
        model = sm.get_model(None, 3, data.get_chunks())
        model.load_state_dict(torch.load(CHECKPOINTS / args.weights))
    
    test_data = Document2Graph('TEST', NAF / 'simple/test', device)
    test_data.get_info()

    test_graphs = dgl.batch(test_data.graphs).to(device)

    model.eval()
    with torch.no_grad():
        scores = model(test_graphs, test_graphs.ndata['feat'].to(device))
        auc = compute_auc_mc(scores.to(device), test_graphs.edata['label'].to(device))
        
        _, preds = torch.max(F.log_softmax(scores, dim=1), dim=1)
        test_graphs.edata['preds'] = preds

        # for g, graph in enumerate(dgl.unbatch(test_graphs)):
        #     targets = graph.edata['preds']
        #     kvp_ids = targets.nonzero().flatten().tolist()
            # test_data.print_graph(num=g, labels_ids=kvp_ids, name=f'test_{g}')
            # test_data.print_graph(num=g, name=f'test_labels_{g}')

        accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graphs.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graphs.edata['label'], per_class=True)
    

    ################* STEP 4: RESULTS ################
    print("\n### RESULTS ###")
    print("AUC {:.4f}".format(auc))
    print("Accuracy {:.4f}".format(accuracy))
    print("F1 Score: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
    print("F1 Classes: None {:.4f} - Key-Value {:.4f} - SameEntity {:.4f}".format(classes_f1[0], classes_f1[1], classes_f1[2]))

    #! test_data.GB.balance_edges(test_data.graphs[test_graph], test_data.edge_num_classes, 0)
    create_folder(OUTPUTS / 'link_predictions')
    for test_graph in range(len(test_data.graphs)):
        ne = test_data.graphs[test_graph].num_edges()
        pne = sum([test_data.graphs[n].num_edges() for n in range(test_graph)])
        _, indices = torch.max(scores, dim=1)
        test_graph_indices = indices[pne:pne+ne]
        from PIL import Image
        img = Image.open(test_data.paths[test_graph]).convert('RGB')
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
            if pred == 0: continue
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