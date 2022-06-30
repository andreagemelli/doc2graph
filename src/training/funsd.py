from asyncore import write
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
import torch
from torch.nn import functional as F
from random import shuffle, choice, seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

from src.data.dataloader import Document2Graph
from src.paths import ROOT, FUNSD_TRAIN, FUNSD_TEST, RUNS, WEIGHTS
from src.training.models import SetModel
from src.amazing_utils import get_config
from src.training.amazing_utils import EarlyStopping, accuracy, compute_auc, compute_auc_2, compute_auc_all, compute_auc_mc, compute_crossentropy_loss, compute_loss_2, evaluate, get_binary_accuracy_and_f1, get_device, get_f1, get_features, save_test_results, validate

def entity_labeling(args):

    # config
    sm = SetModel(name=args.model)
    cfg_train = get_config('train')
    device = get_device(args.gpu)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        data.get_info()
        num_feats = 200 # data.num_features 100
        node_num_classes = data.node_num_classes
        
        #! Not easy
        #TODO change train/val split to be "balanced" according to training
        ss = ShuffleSplit(n_splits=1, test_size=cfg_train.val_size, random_state=0)
        train_index, val_index = next(ss.split(data.graphs)) #? VALIDATION SPLITS
        train_graphs = [data.graphs[i] for i in train_index]
        val_graphs = [data.graphs[i] for i in val_index]

        batch_size = cfg_train.batch_size
        num_batches = int(len(train_graphs) / batch_size)

        ################* STEP 1: CREATE MODEL ################
        model = sm.get_model([node_num_classes, None], data.get_chunks())
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
        model = sm.get_model(num_feats, node_num_classes)
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

def entity_linking(args):
    # configs
    sm = SetModel(name=args.model)
    cfg_train = get_config('train')
    device = get_device(args.gpu)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        data.balance()
        data.get_info()
        
        node_num_feats = data.node_num_features # data.num_features 100
        edge_classes = data.edge_num_classes

        #! Not easy
        #TODO change train/val split to be "balanced" according to training
        ss = ShuffleSplit(n_splits=1, test_size=cfg_train.val_size, random_state=0)
        train_index, val_index = next(ss.split(data.graphs)) #? VALIDATION SPLITS
        # data.balance(indices=train_index)
        train_graphs = [data.graphs[i] for i in train_index]
        val_graphs = [data.graphs[i] for i in val_index]

        batch_size = cfg_train.batch_size
        num_batches = int(len(train_graphs) / batch_size)

        ################* STEP 1: CREATE MODEL ################
        model = sm.get_model([None, edge_classes], data.get_chunks())
        model = model.to(device)
        loss_fcn = torch.nn.CrossEntropyLoss()
        # TODO Try also SGD + CosineAnnealingLR
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
                target = g.edata['label'].to(device)
                logits = model(g, feat)
                loss = loss_fcn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = accuracy(logits, target)
                train_f1, _ = get_f1(logits, target)
                train_acc += acc

            model.eval()
            with torch.no_grad():
                g = dgl.batch(val_graphs)
                g = g.int().to(device)
                feat = g.ndata['feat'].to(device)
                target = g.edata['label'].to(device)
                val_preds = model(g, feat)
                val_acc = accuracy(val_preds, target)
                val_loss = loss_fcn(val_preds, target)
                val_f1, _ = get_f1(val_preds, target)

            scheduler.step(val_loss.item())

            print("Epoch {:05d} | TrainLoss {:.4f} | TrainAcc {:.4f} | ValLoss {:.4f} | ValAcc {:.4f} |"
            .format(epoch, loss.item(), train_acc/batch_size, val_loss.item(), val_acc))
            
            #* UPDATE
            if cfg_train.stopper_metric == 'loss' and stopper.step(val_loss.item()): break
            if cfg_train.stopper_metric == 'acc' and stopper.step(val_acc): break
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        model = sm.get_model(None, [None, edge_classes], data.get_chunks())
        model.load_state_dict(torch.load(ROOT / args.weights))
        model.to(device)
    
    ################* STEP 4: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = device)
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

    ################* STEP 5: RESULTS ################
    print("\n### RESULTS ###")
    print("Mean Test Accuracy {:.4f}".format(test_acc))
    print(f"F1 Score (macro/micro): {macro} {micro}\n F1 per class: {classes_f1}")
    feat_n, feat_e = get_features(args.add_embs, args.add_visual, args.add_eweights)

    #? if skipping training, no need to save anything
    results = {'model': sm.get_name(), 'net-params': sm.get_total_params(), 'features': feat_n, 'fedges': feat_e, 'best_score': stopper.best_score, 'f1-scores': (macro, micro),
            'classes': classes_f1}
    if not args.test: save_test_results(train_name, results)
    return

def link_prediction(args):

    # configs
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        data.get_info()
        data.print_graph()

        ss = ShuffleSplit(n_splits=1, test_size=cfg_train.val_size, random_state=0)
        train_index, val_index = next(ss.split(data.graphs))
        rand_tid = choice(train_index)
        rand_vid = choice(val_index)

        # TRAIN
        train_graphs = [data.graphs[i] for i in train_index]
        tg = dgl.batch(train_graphs)
        tg = tg.int().to(device)
    
        val_graphs = [data.graphs[i] for i in val_index]
        vg = dgl.batch(val_graphs)
        vg = vg.int().to(device)
        
        ################* STEP 1: CREATE MODEL ################
        model = sm.get_model(None, 2, data.get_chunks())
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1000, verbose=True, factor=0.1)
        e = datetime.now()
        train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
        stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=10000)
        writer = SummaryWriter(log_dir=RUNS)
    
        ################* STEP 2: TRAINING ################
        print("\n### TRAINING ###")
        print(f"-> Training samples: {tg.batch_size}")
        print(f"-> Validation samples: {vg.batch_size}\n")

        for epoch in range(cfg_train.epochs):

            #* TRAINING
            model.train()
            
            scores = model(tg, tg.ndata['feat'].to(device))
            # loss = compute_loss_2(scores.flatten().to(device), tg.edata['label'].to(device))
            loss= compute_crossentropy_loss(scores.to(device), tg.edata['label'].to(device))
            auc = compute_auc_mc(scores.to(device), tg.edata['label'].to(device))

            optimizer.zero_grad()
            loss.backward()
            n = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            print("Gradient:", n.max())
            optimizer.step()

            #* VALIDATION
            model.eval()
            with torch.no_grad():
                val_scores = model(vg, vg.ndata['feat'].to(device))
                # val_loss = compute_loss_2(val_scores.flatten().to(device), vg.edata['label'].to(device))
                # val_loss = loss_fcn(val_scores.to(device), vg.edata['label'].to(device))
                val_loss = compute_crossentropy_loss(val_scores.to(device), vg.edata['label'].to(device))
                val_auc = compute_auc_mc(val_scores.to(device), vg.edata['label'].to(device))
            
            #* PRINTING IMAGEs
            
            start, end = 0, 0
            for tid in train_index:
                start = end
                end += data.graphs[tid].num_edges()
                if tid == rand_tid: break

            # targets = torch.where(scores > 0, 1, 0)[start:end]
            blu = F.softmax(scores[start:end], dim=1)
            _, targets = torch.max(blu, dim=1)
            kvp_ids = targets.nonzero().flatten().tolist()
            data.print_graph(num=rand_tid, labels_ids=kvp_ids, name='train')
            data.print_graph(num=rand_tid, name='train_labels')

            v_start, v_end = 0, 0
            for vid in val_index:
                v_start = v_end
                v_end += data.graphs[vid].num_edges()
                if vid == rand_vid: break

            #val_targets = torch.where(scores > 0, 1, 0)[v_start:v_end]
            _, val_targets = torch.max(F.softmax(val_scores[v_start:v_end], dim=1), dim=1)
            val_kvp_ids = val_targets.nonzero().flatten().tolist()
            
            data.print_graph(num=rand_vid, labels_ids=val_kvp_ids, name='val')
            data.print_graph(num=rand_vid, name='val_labels')
            
            scheduler.step(val_loss.item())

            print("Epoch {:05d} | TrainLoss {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValAUC-PR {:.4f} |"
            .format(epoch, loss.item(), auc, val_loss.item(), val_auc))
            
            #* UPDATE
            if cfg_train.stopper_metric == 'loss' and stopper.step(val_loss.item()): break
            if cfg_train.stopper_metric == 'acc' and stopper.step(val_auc): break

            writer.add_scalars('AUC-PR', {'train': auc, 'val': val_auc}, epoch)
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Loss/val', val_loss.item(), epoch)
        
        print("LOADING: ", train_name+'.pt')
        model.load_state_dict(torch.load(WEIGHTS / (train_name+'.pt')))
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        data = Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = device)
        model = sm.get_model(None, 1, data.get_chunks())
        model.load_state_dict(torch.load(WEIGHTS / args.weights))
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = device)
    # test_data.balance()
    test_data.get_info()
    
    test_graph = dgl.batch(test_data.graphs).to(device)
    pos_eids = (test_graph.edata['label'] == 1).nonzero().flatten().tolist()
    test_g_pos = dgl.edge_subgraph(test_graph, pos_eids).to(device)
    neg_eids = (test_graph.edata['label'] == 0).nonzero().flatten().tolist()
    test_g_neg = dgl.edge_subgraph(test_graph, neg_eids).to(device)

    model.eval()
    with torch.no_grad():
        pos_score = model(test_g_pos, test_g_pos.ndata['feat'].to(device))
        neg_score = model(test_g_neg, test_g_neg.ndata['feat'].to(device))
        auc = compute_auc(pos_score, neg_score)

        all_scores = model(test_graph, test_graph.ndata['feat'].to(device))
        out = torch.nn.Sigmoid()
        all_scores = out(all_scores)
        all_auc, ot = compute_auc_all(all_scores, test_graph.edata['label'])

        print(ot[0])
        preds = torch.where(all_scores > ot[0], 1, 0)
        test_graph.edata['preds'] = preds

        accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)
    
    graphs = dgl.unbatch(test_graph)
    target_graph = graphs[0]
    targets = target_graph.edata['preds']
    kvp = (targets == 1)
    kvp_ids = kvp.nonzero().flatten().tolist()
    
    # ! PRINTING IMAGE
    print(test_data.paths[0])
    img = Image.open(test_data.paths[0]).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    boxs = target_graph.ndata['geom'][:, :4].tolist()
    boxs = [[box[0]*w, box[1]*h, box[2]*w, box[3]*h] for box in boxs]
    for box in boxs:
        draw.rectangle(box, outline='blue', width=2)
    
    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
    u, v = target_graph.edges()
    for edge in kvp_ids:
        sb = boxs[u[edge]]
        eb = boxs[v[edge]]
        draw.line((center(sb), center(eb)), fill='violet', width=3)
    img.save('test.png')

    ################* STEP 4: RESULTS ################
    print("\n### RESULTS ###")
    print("AUC {:.4f}".format(auc))
    print("All AUC {:.4f}".format(all_auc))
    print("Accuracy {:.4f}".format(accuracy))
    print("F1 Score: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
    print("F1 Classes: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))

    if not args.test:
        feat_n, feat_e = get_features(args.add_geom, args.add_embs, args.add_visual, args.add_eweights)
        #? if skipping training, no need to save anything
        results = {'model': sm.get_name(), 'net-params': sm.get_total_params(), 'features': feat_n, 'fedges': feat_e, 'best_score': stopper.best_score, 'f1-scores': (f1[0], f1[1]),
                'classes': classes_f1}
        save_test_results(train_name, results)
    return


def train_funsd(args):

    if args.task == 'elab':
        entity_labeling(args)
    elif args.task == 'elin':
        # entity_linking(args)
        link_prediction(args)
    elif args.task == 'wgrp':
        word_grouping(args)
    else:
        raise Exception("Task selected does not exists. Enter:\
            - 'elab': entity labeling\
            - 'elin': entity linking\
            - 'wgrp': word grouping")
    return

#! OLD CODE
# ! OLD BINARY LINK PREDICTION APPROACH
        # pos_eids = (train_graph.edata['label'] == 1).nonzero().flatten().tolist()
        # train_g_pos = dgl.edge_subgraph(train_graph, pos_eids)
        # neg_eids = (train_graph.edata['label'] == 0).nonzero().flatten().tolist()
        # train_g_neg = dgl.edge_subgraph(train_graph, neg_eids)
# ! OLD BINARY LINK PREDICTION APPROACH
        # pos_eids = (val_graph.edata['label'] == 1).nonzero().flatten().tolist()
        # val_g_pos = dgl.edge_subgraph(val_graph, pos_eids)
        # neg_eids = (val_graph.edata['label'] == 0).nonzero().flatten().tolist()
        # val_g_neg = dgl.edge_subgraph(val_graph, neg_eids)

        # neg_feat = train_g_neg.ndata['feat'].to(device)
        # neg_score = model(train_g_neg, neg_feat)

        # val_pos_score = model(val_g_pos, val_g_pos.ndata['feat'].to(device))
        # val_neg_score = model(val_g_neg, val_g_neg.ndata['feat'].to(device))
        # val_loss = compute_loss(val_pos_score, val_neg_score, device)
        # val_auc = compute_auc(val_pos_score, val_neg_score)