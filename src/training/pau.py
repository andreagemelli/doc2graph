from datetime import datetime
from sklearn.model_selection import KFold, ShuffleSplit
import torch
from torch.nn import functional as F
from random import shuffle, seed
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import dgl 
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
from knockknock import discord_sender
from statistics import mean
import numpy as np

from src.data.dataloader import Document2Graph
from src.paths import *
from src.models.graphs import SetModel
from src.amazing_utils import get_config
from src.training.amazing_utils import *


webhook_url = "https://discord.com/api/webhooks/997898000695316581/mIlHnPn95KTOYPS-hQFQFwaEhk0pa82ZR12OMZWZqXWwlH7fHjMZGnCXa03nvO_4cMzw"
@discord_sender(webhook_url=webhook_url)
def train_pau(args):

    # configs
    start_training = time.time()
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='PAU TRAIN', src_path=PAU_TRAIN, device = device, output_dir=TRAIN_SAMPLES)
        data.get_info()

        # ss = ShuffleSplit(n_splits=10, test_size=cfg_train.val_size, random_state=cfg_train.seed)
        ss = KFold(n_splits=7, shuffle=True, random_state=cfg_train.seed)
        cv_indices = ss.split(data.graphs)
        
        models = []
        train_index, val_index = next(ss.split(data.graphs))
        
        for cvs in cv_indices:

            train_index, val_index = cvs

            # TRAIN
            train_graphs = [data.graphs[i] for i in train_index]
            tg = dgl.batch(train_graphs)
            tg = tg.int().to(device)
        
            val_graphs = [data.graphs[i] for i in val_index]
            vg = dgl.batch(val_graphs)
            vg = vg.int().to(device)
            
            ################* STEP 1: CREATE MODEL ################
            model = sm.get_model(data.node_num_classes, data.edge_num_classes, data.get_chunks())
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
            # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=400, min_lr=1e-3, verbose=True, factor=0.01)
            # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            e = datetime.now()
            train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
            models.append(train_name+'.pt')
            stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=2000)
            # writer = SummaryWriter(log_dir=RUNS)
            # convert_imgs = transforms.ToTensor()
        
            ################* STEP 2: TRAINING ################
            print("\n### TRAINING ###")
            print(f"-> Training samples: {tg.batch_size}")
            print(f"-> Validation samples: {vg.batch_size}\n")

            # im_step = 0
            for epoch in range(cfg_train.epochs):

                #* TRAINING
                model.train()
                
                n_scores, e_scores = model(tg, tg.ndata['feat'].to(device))
                # loss = compute_loss_2(scores.flatten().to(device), tg.edata['label'].to(device))
                n_loss = compute_crossentropy_loss(n_scores.to(device), tg.ndata['label'].to(device))
                e_loss = compute_crossentropy_loss(e_scores.to(device), tg.edata['label'].to(device))
                tot_loss = n_loss + e_loss

                macro, micro = get_f1(n_scores, tg.ndata['label'].to(device))
                auc = compute_auc_mc(e_scores.to(device), tg.edata['label'].to(device))

                optimizer.zero_grad()
                tot_loss.backward()
                n = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                #* VALIDATION
                model.eval()
                with torch.no_grad():
                    val_n_scores, val_e_scores = model(vg, vg.ndata['feat'].to(device))
                    val_n_loss = compute_crossentropy_loss(val_n_scores.to(device), vg.ndata['label'].to(device))
                    val_e_loss = compute_crossentropy_loss(val_e_scores.to(device), vg.edata['label'].to(device))
                    val_tot_loss = val_n_loss + val_e_loss
                    val_macro, val_micro = get_f1(val_n_scores, vg.ndata['label'].to(device))
                    val_auc = compute_auc_mc(val_e_scores.to(device), vg.edata['label'].to(device))
                
                # scheduler.step(val_auc)
                # scheduler.step()

                #* PRINTING IMAGEs AND RESULTS

                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} | ValAUC-PR {:.4f} |"
                .format(epoch, tot_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))
                
                if cfg_train.stopper_metric == 'loss':
                    step_value = val_tot_loss.item()
                elif cfg_train.stopper_metric == 'acc':
                    step_value = val_micro
                
                #if val_auc > best_val_auc:
                #    best_val_auc = val_auc
                #    best_model = train_name
                
                ss = stopper.step(step_value)

                # if  ss == 'improved':
                #     im_step = epoch
                #     train_imgs = []
                #     for r in rand_tid:
                #         start, end = 0, 0
                #         for tid in train_index:
                #             start = end
                #             end += data.graphs[tid].num_edges()
                #             if tid == r: break

                #         _, targets = torch.max(F.log_softmax(e_scores[start:end], dim=1), dim=1)
                #         kvp_ids = targets.nonzero().flatten().tolist()
                #         train_imgs.append(convert_imgs(data.print_graph(num=r, labels_ids=kvp_ids, name=f'train_{r}'))[:, :, :700])
                #         # data.print_graph(num=r, name=f'train_labels_{r}')

                #     val_imgs = []
                #     for r in rand_vid:
                #         v_start, v_end = 0, 0
                #         for vid in val_index:
                #             v_start = v_end
                #             v_end += data.graphs[vid].num_edges()
                #             if vid == r: break

                #         _, val_targets = torch.max(F.log_softmax(val_e_scores[v_start:v_end], dim=1), dim=1)
                #         val_kvp_ids = val_targets.nonzero().flatten().tolist()
                #         val_imgs.append(convert_imgs(data.print_graph(num=r, labels_ids=val_kvp_ids, name=f'val_{r}'))[:, :, :700])
                        # data.print_graph(num=r, name=f'val_labels_{r}')

                if ss == 'stop':
                    break

                # writer.add_scalars('AUC-PR', {'train': auc, 'val': val_auc}, epoch)
                # writer.add_scalars('LOSS', {'train': tot_loss.item(), 'val': val_tot_loss.item()}, epoch)
                # writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

                # train_grid = torchvision.utils.make_grid(train_imgs)
                # writer.add_image('train_images', train_grid, im_step)
                # val_grid = torchvision.utils.make_grid(val_imgs)
                # writer.add_image('val_images', val_grid, im_step)
        
        # print("LOADING: ", best_model+'.pt')
        # model.load_state_dict(torch.load(WEIGHTS / (best_model+'.pt')))
    
    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        models = args.weights
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Document2Graph(name='PAU TEST', src_path=PAU_TEST, device = device, output_dir=TEST_SAMPLES)
    test_data.get_info()
    model = sm.get_model(test_data.node_num_classes, test_data.edge_num_classes, test_data.get_chunks())
    best_model = ''
    nodes_micro = []
    edges_f1 = []
    test_graph = dgl.batch(test_data.graphs).to(device)

    for m in models:
        model.load_state_dict(torch.load(CHECKPOINTS / m))
        model.eval()
        with torch.no_grad():

            n, e = model(test_graph, test_graph.ndata['feat'].to(device))
            auc = compute_auc_mc(e.to(device), test_graph.edata['label'].to(device))
            _, preds = torch.max(F.softmax(e, dim=1), dim=1)

            accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
            _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)
            edges_f1.append(classes_f1[1])

            macro, micro = get_f1(n, test_graph.ndata['label'].to(device))
            nodes_micro.append(micro)
            if micro >= max(nodes_micro):
                best_model = m

        ################* STEP 4: RESULTS ################
        print("\n### RESULTS ###")
        print("AUC {:.4f}".format(auc))
        print("Accuracy {:.4f}".format(accuracy))
        print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
        print("F1 Edges: None {:.4f} - Table {:.4f}".format(classes_f1[0], classes_f1[1]))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))
    
    print(f"\nLoading best model {best_model}")
    model.load_state_dict(torch.load(CHECKPOINTS / best_model))
    model.eval()
    with torch.no_grad():

        n, e= model(test_graph, test_graph.ndata['feat'].to(device))
        # auc = compute_auc_mc(scores.to(device), test_graph.edata['label'].to(device))
        
        _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
        _, npreds = torch.max(F.softmax(n, dim=1), dim=1)

        accuracy, f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'], per_class=True)
        macro, micro = get_f1(n, test_graph.ndata['label'].to(device))
    
    test_graph.edata['preds'] = epreds
    test_graph.ndata['preds'] = npreds
    for g, graph in enumerate(dgl.unbatch(test_graph)):
        etargets = graph.edata['preds']
        ntargets = graph.ndata['preds']
        kvp_ids = etargets.nonzero().flatten().tolist()
        test_data.print_graph(num=g, labels_ids=kvp_ids, name=f'test_{g}')
        test_data.print_graph(num=g, name=f'test_labels_{g}')

    if not args.test:
        feat_n, feat_e = get_features(args)
        #? if skipping training, no need to save anything
        model = get_config(CFGM / args.model)
        results = {'MODEL': {
            'name': sm.get_name(),
            'weights': best_model,
            'net-params': sm.get_total_params(), 
            'num-layers': model.num_layers,
            'projector-output': model.out_chunks,
            'dropout': model.dropout,
            'lastFC': model.hidden_dim
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'RESULTS': {
                'val-loss': stopper.best_score, 
                'f1-scores': f1,
		        'f1-classes': classes_f1,
                'nodes-f1': [macro, micro],
                'std-pairs': np.std(nodes_micro),
                'mean-pairs': mean(nodes_micro)
            }}
        save_test_results(train_name, results)
    
    print("END TRAINING:", time.time() - start_training)
    return {'best_model': best_model, 'Nodes-F1': {'max': max(nodes_micro), 'mean': mean(nodes_micro), 'std': np.std(nodes_micro)}, 'Edges-F1': {'max': max(edges_f1), 'mean': mean(edges_f1), 'std': np.std(edges_f1)}}
    # # TODO: FIX and CHECK
    # pairs_ids = preds.nonzero().flatten().tolist()
    # u, v = test_graph.edges()

    # for idd, dst in enumerate(v.tolist()):
    #     if idd not in pairs_ids: continue
    #     scores_ = [scores[idd]]
    #     id_ = [idd]
    #     for ido, oth in enumerate(v.tolist()):
    #         if ido != idd and dst == oth:
    #             id_.append(ido)
    #             scores_.append(scores[ido])
    #     id_.pop(scores_.index(max(scores_)))
    #     for id in id_: preds[id] = 0
    
    # test_graph.edata['preds'] = preds

    # accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
    # _, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)

    # macro, micro = get_f1(n, test_graph.ndata['label'].to(device))

    # ################* STEP 4: RESULTS ################
    # print("\n### RESULTS POST-PROCESSING###")
    # print("AUC {:.4f}".format(auc))
    # print("Accuracy {:.4f}".format(accuracy))
    # print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
    # print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
    # print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    if not args.test:
        feat_n, feat_e = get_features(args)
        #? if skipping training, no need to save anything
        model = get_config(CFGM / args.model)
        results = {'MODEL': {
            'name': sm.get_name(), 
            'net-params': sm.get_total_params(),
            'projector-output': model.out_chunks,
            'dropout': model.dropout,
            'lastFC': model.hidden_dim
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'RESULTS': {
                'val-loss': stopper.best_score, 
                'f1-edge': f1,
		        'f1-edge': classes_f1,
                'f1-nodes': [macro, micro],
                'AUC-PR': auc,
            }}
        save_test_results(train_name, results)
    
    print("END TRAINING:", time.time() - start_training)
    return
