from attrdict import AttrDict
from sklearn.model_selection import ShuffleSplit
import torch
import yaml
from random import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau

from math import inf
import dgl

from src.data.dataloader import DocumentGraphs
from src.paths import ROOT, TEST, TRAIN, CONFIGS
from src.utils.train import EarlyStopping, accuracy, evaluate, get_model, get_device, save_best_results, save_test_results, validate

def entity_labeling(args):

    # configs
    with open(CONFIGS / f'{args.config}.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
    device = get_device(args.gpu)

    ################* STEP 0: LOAD DATA ################
    data = DocumentGraphs(name='FUNSD TRAIN', src_path=TRAIN, add_embs=args.add_embs)
    print()
    data.get_info()
    n_classes = data.num_classes
    num_feats = data.num_features

    ################* STEP 1: TRAINING ################
    print("\n### TRAINING ###")
    batch_sizes = [1, int(len(data.graphs)/10)] #? grid search over batch size
    num_layers = [2, 3, 5] #? grid search over network depth
    best_params = {'val_loss': inf} #? best params
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    split = 0

    for train_index, val_index in ss.split(data.graphs): #? VALIDATION SPLITS

        train_graphs = [data.graphs[i] for i in train_index]
        if len(train_graphs) not in batch_sizes: batch_sizes.append(len(train_graphs))
        val_graphs = [data.graphs[i] for i in val_index]

        for bs in batch_sizes: #? GRID #1
            for nl in num_layers: #? GRID #2

                mc = {'name': config.MODEL.name, 'num_layers': nl, 'hidden_dim': 64, 'dropout':0.}
                model, tp = get_model(mc, [num_feats, n_classes], args.add_attn)
                model = model.to(device)
                loss_fcn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.lr, weight_decay=config.TRAIN.weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.5)
                stopper = EarlyStopping(model, cv=True)

                for epoch in range(config.TRAIN.epochs):
                    model.train()
                    train_acc = 0
                    shuffle(train_graphs)

                    for b in range(int(len(train_graphs) / bs)):
                        train_batch = train_graphs[b * bs: min((b+1)*bs, len(train_graphs))]
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

                    print("Split {}/{} | Epoch {:05d} | TrainLoss {:.4f} | TrainAcc {:.4f} | ValLoss {:.4f} | ValAcc {:.4f} |"
                    .format(split + 1, ss.n_splits, epoch, loss.item(), mean_train_acc, val_loss, val_acc))
                    
                    #* UPDATE
                    if stopper.step(val_loss, model): break
                    scheduler.step(val_loss)
                    if val_loss < best_params['val_loss']:
                        best_params['val_loss'] = val_loss
                        best_params['batch_size'] = bs
                        best_params['num_layers'] = nl
                        best_params['split'] = split
                        best_params['model'] = f'{stopper.name}.pt'
                        best_params['total_params'] = tp

        split += 1
    
    ################* STEP 3: SAVING BEST MODEL ################
    print("Best Paramas:", best_params)
    save_best_results(best_params, rm_logs=True)
    
    ################* STEP 4: TESTING ################
    print("\n### TESTING ###")

    #? load best model
    best_model = best_params['model'].split(".")[0]
    best_model_weights = ROOT / 'output' / 'weights' / best_params['model']
    with open(CONFIGS / f'{best_model}.yaml') as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
    model, tp = get_model(config.MODEL, [num_feats, n_classes], args.add_attn)
    model.load_state_dict(torch.load(best_model_weights))
    model.to(device)

    #? test
    test_data = DocumentGraphs(name='FUNSD TEST', src_path=TEST, add_embs=args.add_embs)
    test_data.get_info()
    mean_test_acc, f1 = evaluate(model, test_data.graphs, device)
    print("Mean Test Accuracy {:.4f}".format(mean_test_acc))
    print(f"F1 Score (macro/micro): {f1}")

    save_test_results(best_params['model'].split(".")[0], {'features': args.add_embs, 'f1': f1})
    return

def word_grouping():
    #TODO
    return

def entity_linking():
    #TODO
    return