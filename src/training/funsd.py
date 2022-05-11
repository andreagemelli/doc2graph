from datetime import datetime
from sklearn.model_selection import ShuffleSplit
import torch
from random import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl

from src.data.dataloader import Document2Graph
from src.paths import ROOT, FUNSD_TRAIN, FUNSD_TEST
from src.training.models import SetModel
from src.utils import get_config
from src.training.utils import EarlyStopping, accuracy, evaluate, get_device, get_features, save_test_results, validate

def entity_labeling(args):

    # configs
    sm = SetModel(name=args.model)
    cfg_train = get_config('train')
    device = get_device(args.gpu)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        data.get_info()
        num_feats = 200 # data.num_features 100
        num_classes = data.num_classes
        
        #! Not easy
        #TODO change train/val split to be "balanced" according to training
        ss = ShuffleSplit(n_splits=1, test_size=cfg_train.val_size, random_state=0)
        train_index, val_index = next(ss.split(data.graphs)) #? VALIDATION SPLITS
        train_graphs = [data.graphs[i] for i in train_index]
        val_graphs = [data.graphs[i] for i in val_index]

        batch_size = cfg_train.batch_size
        num_batches = int(len(train_graphs) / batch_size)

        ################* STEP 1: CREATE MODEL ################
        model = sm.get_model(num_feats, num_classes)
        model = model.to(device)
        loss_fcn = torch.nn.CrossEntropyLoss()
        #TODO Try also SGD + CosineAnnealingLR
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.05)
        e = datetime.now()
        train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
        stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric)
    
        ################* STEP 1: TRAINING ################
        print("\n### TRAINING ###")
        print(f"-> Training samples: {len(train_graphs)}")
        print(f"-> Validation samples: {len(val_graphs)}\n")

        for epoch in range(cfg_train.epochs):
            model.train()
            train_acc = 0
            shuffle(train_graphs)

            for b in range(num_batches + 1):
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

            mean_train_acc = train_acc / (num_batches + 1)
            val_acc, val_loss = validate(model, val_graphs, device, loss_fcn)
            scheduler.step(val_loss)

            print("Epoch {:05d} | TrainLoss {:.4f} | TrainAcc {:.4f} | ValLoss {:.4f} | ValAcc {:.4f} |"
            .format(epoch, loss.item(), mean_train_acc, val_loss, val_acc))
            
            #* UPDATE
            if cfg_train.stopper_metric == 'loss' and stopper.step(val_loss): break
            if cfg_train.stopper_metric == 'acc' and stopper.step(val_acc): break
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        model = sm.get_model(num_feats, num_classes)
        model.load_state_dict(torch.load(ROOT / args.weights))
        model.to(device)
    
    ################* STEP 4: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = device)
    test_data.get_info()
    mean_test_acc, f1 = evaluate(model, test_data.graphs, device)

    ################* STEP 5: RESULTS ################
    print("\n### RESULTS ###")
    print("Mean Test Accuracy {:.4f}".format(mean_test_acc))
    print(f"F1 Score (macro/micro): {f1}")
    feat_n, feat_e = get_features(args.add_embs, args.add_visual, args.add_eweights)

    #? if skipping training, no need to save anything
    results = {'model': sm.get_name(), 'net-params': sm.get_total_params(), 'features': feat_n, 'fedges': feat_e, 'val-loss': stopper.best_score, 'f1-scores': f1}
    if not args.test: save_test_results(train_name, results)
    return

def word_grouping():
    #TODO
    return

def entity_linking():
    #TODO
    return

def train_funsd(args):

    if args.task == 'elab':
        entity_labeling(args)
    elif args.task == 'elin':
        entity_linking(args)
    elif args.task == 'wgrp':
        word_grouping(args)
    else:
        raise Exception("Task selected does not exists. Enter:\
            - 'elab': entity labeling\
            - 'elin': entity linking\
            - 'wgrp': word grouping")
    return