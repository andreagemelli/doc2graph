import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn import E2E
from src.models.utils import compute_auc_mc, compute_crossentropy_loss, get_binary_accuracy_and_f1, get_f1
from src.utils import get_config
import src.globals as glb


class Doc2GraphModel():

    def __init__(self, nodes, edges, chunks):
        """ Create a SetModel object, that handles dinamically different version of Doc2Graph Model. Default "end-to-end" (e2e)

        Args:
            name (str) : Which model to train / test. Default: e2e [gcn, edge].
        
        Returns:
            SetModel object.
        """

        self.cfg_model = get_config('model')
        self.total_params = 0
        self.device = glb.DEVICE
        self.model = self.__set_model(nodes, edges, chunks)
        self.get_info()
    
    def __set_model(self, nodes : int, edges : int, chunks : list) -> nn.Module:
        """Return the DGL model defined in the setting file

        Args:
            nodes (int) : number of nodes target class
            edges (int) : number of edges target class
            chunks (list) : list of indeces of chunks

        Returns:
            A PyTorch nn.Module, your DGL model.
        """
        print("\n### MODEL ###")

        edge_pred_features = int((math.log2(get_config('preprocessing').FEATURES.num_polar_bins) + nodes)*2)
        m = E2E(nodes, edges, chunks, self.device,  edge_pred_features,
                self.cfg_model.num_layers, self.cfg_model.dropout,  self.cfg_model.out_chunks, self.cfg_model.hidden_dim,  self.cfg_model.doProject)
        
        m.to(self.device)
        self.total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

        return m
    
    def get_info(self) -> None:
        """ Printing info utility
        """
        print(f"-> Total params: {self.total_params}")
        print("-> Device: " + str(next(self.model.parameters()).is_cuda) + "\n")
        print(self.model)
    
    def get_total_params(self) -> int:
        """ Returns number of model parameteres.
        """
        return self.total_params
    
    def set_optimizer(self, lr, wd):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
    
    def train(self, epoch, tg, vg):
        # TODO insert training with batch

        #* TRAIN
        self.model.train()

        n_scores, e_scores = self.model(tg, tg.ndata['feat'].to(self.device))
        n_loss = compute_crossentropy_loss(n_scores.to(self.device), tg.ndata['label'].to(self.device))
        e_loss = compute_crossentropy_loss(e_scores.to(self.device), tg.edata['label'].to(self.device))
        tot_loss = n_loss + e_loss
        macro, micro = get_f1(n_scores, tg.ndata['label'].to(self.device))
        auc = compute_auc_mc(e_scores.to(self.device), tg.edata['label'].to(self.device))

        self.optimizer.zero_grad()
        tot_loss.backward()
        n = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        #* VALIDATION
        self.model.eval()
        with torch.no_grad():
            val_n_scores, val_e_scores = self.model(vg, vg.ndata['feat'].to(self.device))
            val_n_loss = compute_crossentropy_loss(val_n_scores.to(self.device), vg.ndata['label'].to(self.device))
            val_e_loss = compute_crossentropy_loss(val_e_scores.to(self.device), vg.edata['label'].to(self.device))
            val_tot_loss = val_n_loss + val_e_loss
            val_macro, _ = get_f1(val_n_scores, vg.ndata['label'].to(self.device))
            val_auc = compute_auc_mc(val_e_scores.to(self.device), vg.edata['label'].to(self.device))

        print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} | ValAUC-PR {:.4f} |"
        .format(epoch, tot_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))
        
        return val_tot_loss, val_auc
    
    def test(self, tg):
        self.model.eval()
        with torch.no_grad():

            n, e= self.model(tg, tg.ndata['feat'].to(self.device))
            auc = compute_auc_mc(e.to(self.device), tg.edata['label'].to(self.device))     
            _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
            _, npreds = torch.max(F.softmax(n, dim=1), dim=1)

            accuracy, f1 = get_binary_accuracy_and_f1(epreds, tg.edata['label'])
            _, classes_f1 = get_binary_accuracy_and_f1(epreds, tg.edata['label'], per_class=True)
            macro, micro = get_f1(n, tg.ndata['label'].to(self.device))

            # TODO: check these variables
            tg.edata['preds'] = epreds
            tg.ndata['preds'] = npreds
            tg.ndata['net'] = n

        return classes_f1, (macro, micro)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    

