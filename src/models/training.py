from sklearn.model_selection import KFold
from random import seed
import dgl 
import time
from PIL import Image

from src.data.dataloader import Doc2GraphLoader
from src.paths import *
from src.models.setter import Doc2GraphModel
from src.utils import get_config
from src.models.utils import *
import src.globals as glb

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

        tg_bboxs, tg_images = data.get_bboxs_and_imgs(train_index)
    
        val_graphs = [data.graphs[i] for i in val_index]
        vg = dgl.batch(val_graphs)
        vg = vg.int().to(glb.DEVICE)
        vg_bboxs, vg_images = data.get_bboxs_and_imgs(val_index)
        
        model.set_optimizer(lr=float(cfg_train.lr), wd=float(cfg_train.weight_decay))
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=400, min_lr=1e-3, verbose=True, factor=0.01)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        models.append(f"{run}/split{split}.pt")
        stopper = EarlyStopping(model, f"{run}/split{split}.pt", metric=cfg_train.stopper_metric, patience=2000)
    
        print(f"\n### TRAINING - SPLIT {split} ###")
        print(f"-> Training samples: {len(train_index)}")
        print(f"-> Validation samples: {vg.batch_size}\n")

        #!! NEW CODE FOR VISUAL EMBEDDER

        k = len(train_index)
        bs = cfg_train.batch_size
        
        for epoch in range(cfg_train.epochs):

            #* TRAINING with mini-batch
            nb = int(k/bs) + 1
            for n in range(nb):
                print("Batch {} - {} : {}".format(n, n*bs, min(n*bs + bs, k)), end='\r')
                train_graphs = [data.graphs[i] for i in train_index[n*bs : min(n*bs + bs, k)]]
                tg = dgl.batch(train_graphs)
                tg = tg.int().to(glb.DEVICE)
                train_tot_loss, train_auc = model.train(tg, tg_images[n*bs : min(n*bs + bs, k)], tg_bboxs[n*bs : min(n*bs + bs, k)])

            val_tot_loss, val_auc = model.validate(vg, vg_images, vg_bboxs)

            print("Epoch {:05d} | TrainLoss {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValAUC-PR {:.4f} |"
                .format(epoch, train_tot_loss.item(), train_auc, val_tot_loss.item(), val_auc))

            if cfg_train.stopper_metric == 'loss':
                step_value = val_tot_loss
            else:
                step_value = val_auc
            
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

    print(f"END TRAINING: {round(time.time() - start_training, 2)}s")
    return run
