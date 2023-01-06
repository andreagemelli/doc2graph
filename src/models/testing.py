from random import seed
import dgl
from statistics import mean
import numpy as np

from src.data.dataloader import Doc2GraphLoader
from src.paths import *
from src.models.setter import Doc2GraphModel
from src.utils import get_config
from src.models.utils import *
import src.globals as glb

def test(args, src):

    # configs
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    run_testing = False

    if 'run' in args.weights:
        run_testing = True
        run = args.weights
        models = [f'{run}/{s}' for s in os.listdir(CHECKPOINTS / run)]
    else:
        models = args.weights
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")

    #? test
    test_data = Doc2GraphLoader(name='TEST', src_path=src, output_dir=TEST_SAMPLES)
    test_data.get_info()
    test_graph = dgl.batch(test_data.graphs).to(glb.DEVICE)
    
    model = Doc2GraphModel(test_data.node_num_classes, test_data.edge_num_classes, test_data.get_chunks())
    
    if run_testing:
        best_model = ''
        nodes_micro = []
        edges_f1 = []

        for m in models:
            model.load(CHECKPOINTS / m)
            edge_f1, node_f1 = model.test(test_graph)
                
            edges_f1.append(edge_f1[1])
            nodes_micro.append(node_f1[1])

            if edge_f1[1] >= max(edges_f1):
                best_model = m

            ################* STEP 4: RESULTS ################
            print("\n### RESULTS {} ###".format(m))
            print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(edge_f1[0], edge_f1[1]))
            print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(node_f1[0], node_f1[1]))
    else:
        best_model = models

    print(f"\n -> Loading {best_model}")
    model.load(CHECKPOINTS / best_model)
    edge_f1, node_f1 = model.test(test_graph)

    # ################* STEP 4: RESULTS ################
    print("\n### BEST RESULTS ###")
    print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(edge_f1[0], edge_f1[1]))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(node_f1[0], node_f1[1]))

    if run_testing:
        print("\n### AVG RESULTS ###")
        print("Semantic Entity Labeling: MEAN ", mean(nodes_micro), " STD: ", np.std(nodes_micro))
        print("Entity Linking: MEAN ", mean(edges_f1),"STD", np.std(edges_f1))

    if not args.test:
        feat_n, feat_e = get_features(args)
        results = {
            'weights': best_model,
            'net-params': model.get_total_params(),
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'NODES': {
                'nodes-f1 [macro, micro]': node_f1,
                'std': np.std(nodes_micro),
                'mean': mean(nodes_micro)
            },
            'EDGES': {
		        'edges-f1 [none, pairs]': edge_f1,
                'std-pairs': np.std(edges_f1),
                'mean-pairs': mean(edges_f1)
            }}

        save_test_results(best_model, results)
    
    return
