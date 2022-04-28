import argparse
from datetime import datetime
import os
from attrdict import AttrDict
from sklearn.model_selection import ShuffleSplit
import torch
import yaml
from random import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl

from src.data.dataloader import DocumentGraphs
from src.paths import CONFIGS, ROOT, FUNSD_TRAIN, FUNSD_TEST
from src.utils.common import create_folder
from src.utils.train import EarlyStopping, accuracy, evaluate, get_model, get_device, save_test_results, validate

def entity_labeling(args):

    # configs
    with open(CONFIGS / f'{args.config}.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
    device = get_device(args.gpu)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = DocumentGraphs(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, add_embs=args.add_embs)
        data.get_info()
        n_classes = data.num_classes
        num_feats = data.num_features

        ################* STEP 1: CREATE MODEL ################
        model, tp = get_model(config.MODEL, [num_feats, n_classes], args.add_attn)
        model = model.to(device)
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.lr, weight_decay=config.TRAIN.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.1)
        train_name = args.config + '-' + str(datetime.timestamp(datetime.now())).split(".")[0]
        stopper = EarlyStopping(model, name=train_name)

    
        ################* STEP 1: TRAINING ################
        print("\n### TRAINING ###")
        batch_size = config.TRAIN.batch_size
        ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

        for train_index, val_index in ss.split(data.graphs): #? VALIDATION SPLITS

            train_graphs = [data.graphs[i] for i in train_index]
            val_graphs = [data.graphs[i] for i in val_index]

            for epoch in range(config.TRAIN.epochs):
                model.train()
                train_acc = 0
                shuffle(train_graphs)

                for b in range(int(len(train_graphs) / batch_size)):
                    train_batch = train_graphs[b * batch_size: min((b+1)*batch_size, len(train_graphs))]
                    g = dgl.batch(train_batch)
                    g = g.int().to(device)
                    feat = g.ndata['feat'].to(device)
                    target = g.ndata['label'].to(device)
                    logits, attn = model(g, feat)
                    loss = loss_fcn(logits, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    acc = accuracy(logits, target)
                    train_acc += acc

                mean_train_acc = train_acc / len(data.graphs)
                val_acc, val_loss = validate(model, val_graphs, device, loss_fcn)
                scheduler.step(val_loss)

                print("Epoch {:05d} | TrainLoss {:.4f} | TrainAcc {:.4f} | ValLoss {:.4f} | ValAcc {:.4f} |"
                .format(epoch, loss.item(), mean_train_acc, val_loss, val_acc))
                
                #* UPDATE
                if stopper.step(val_loss, model): break
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f" -> loading {args.weights}")
        data = DocumentGraphs(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, add_embs=args.add_embs)
        model, tp = get_model(config.MODEL, [data.num_features, data.num_classes], args.add_attn)
        model.load_state_dict(torch.load(ROOT / args.weights))
        model.to(device)
    
    ################* STEP 4: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = DocumentGraphs(name='FUNSD TEST', src_path=FUNSD_TEST, add_embs=args.add_embs)
    test_data.get_info()
    mean_test_acc, f1 = evaluate(model, test_data.graphs, device)

    ################* STEP 5: RESULTS ################
    print("\n### RESULTS ###")
    print("Mean Test Accuracy {:.4f}".format(mean_test_acc))
    print(f"F1 Score (macro/micro): {f1}")
    if args.add_embs: feat = 'text'
    else: feat = 'bbox'

    #? if skipping training, no need to save anything
    results = {'model': config.MODEL.name, 'net-params': tp, 'features': feat, 'val-loss': stopper.best_val_loss, 'f1-scores': f1}
    if not args.test: save_test_results(train_name, results)
    return

def word_grouping():
    #TODO
    return

def entity_linking():
    #TODO
    return

def train(args):
    
    create_folder('output')
    create_folder('output/weights')
    create_folder('output/runs')

    if args.task == 'elab':
        entity_labeling(args)
    elif args.task == 'elin':
        entity_linking(args)
    elif args.task == 'wgrp':
        word_grouping(args)
    else:
        raise "Task selected does not exists. Enter:\
            - 'elab': entity labeling\
            - 'elin': entity linking\
            - 'wgrp': word grouping"
    return