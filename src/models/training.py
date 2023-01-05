from sklearn.model_selection import KFold
from random import seed
import dgl 
import time

from src.data.dataloader import Doc2GraphLoader
from src.paths import *
from src.models.setter import Doc2GraphModel
from src.utils import get_config
from src.models.utils import *
from src.globals import DEVICE

def train(src):

    # configs
    cfg_train = get_config('train')
    seed(cfg_train.seed)

    ################* STEP 1: LOAD DATA ################
    data = Doc2GraphLoader(name='TRAIN', src_path=src, output_dir=TRAIN_SAMPLES)
    data.get_info()
    
    run = create_run_folder()
    models = []

    ss = KFold(n_splits=cfg_train.n_splits, shuffle=True, random_state=cfg_train.seed)
    cv_indices = ss.split(data.graphs)
    train_index, val_index = next(ss.split(data.graphs))

    start_training = time.time()
    for split, cvs in enumerate(cv_indices):

        ################* STEP 2: SET MODEL ################
        model = Doc2GraphModel(data.node_num_classes, data.edge_num_classes, data.get_chunks())

        train_index, val_index = cvs

        train_graphs = [data.graphs[i] for i in train_index]
        tg = dgl.batch(train_graphs)
        tg = tg.int().to(DEVICE)
    
        val_graphs = [data.graphs[i] for i in val_index]
        vg = dgl.batch(val_graphs)
        vg = vg.int().to(DEVICE)
        
        model.set_optimizer(lr=float(cfg_train.lr), wd=float(cfg_train.weight_decay))
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=400, min_lr=1e-3, verbose=True, factor=0.01)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        models.append(f"{run}/split{split}.pt")
        stopper = EarlyStopping(model, f"{run}/split{split}.pt", metric=cfg_train.stopper_metric, patience=2000)
    
        print(f"\n### TRAINING - SPLIT {split}###")
        print(f"-> Training samples: {tg.batch_size}")
        print(f"-> Validation samples: {vg.batch_size}\n")

        # im_step = 0
        for epoch in range(cfg_train.epochs):

            #* TRAINING
            loss, acc = model.train(epoch)

            if cfg_train.stopper_metric == 'loss':
                step_value = loss
            else:
                step_value = acc
            
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

    print("END TRAINING:", time.time() - start_training)
