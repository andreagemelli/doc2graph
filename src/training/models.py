import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
import math
import torch.nn.functional as F
import torchvision

from src.paths import MODELS
from src.utils import get_config

class SetModel():
    def __init__(self, name='gcn') -> None:

        self.cfg_model = get_config(MODELS / name)
        self.name = self.cfg_model.name
        self.total_params = 0
    
    def get_name(self):
        return self.name
    
    def get_total_params(self):
        return self.total_params

    def get_model(self, num_features, num_classes):
        """Return the DGL model defined in the setting file

        Args:
            nums (list): num_features and num_classes

        Returns:
            A PyTorch Model
        """
        print("\n### MODEL ###")
        print(f"-> Using {self.name}")

        if self.name == 'GCN':
            m = GcnSAGE(num_features, self.cfg_model.hidden_dim, num_classes, self.cfg_model.num_layers, F.relu, self.cfg_model.dropout, self.cfg_model.attn)
            self.total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"-> Total params: {self.total_params}\n")
            print(m)
            return m

        elif self.name == 'GAT':
            return ""

        else:
            raise Exception(f"Error! Model {self.name} do not exists.")

class GAT(nn.Module):

    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, h):
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits, a = self.gat_layers[-1](g, h, True)
        return logits.mean(1), a

class GcnSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GcnSAGELayer, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        
        if not self.use_pp:
            norm = self.get_norm(g)
            g.ndata['h'] = h
            g.update_all(fn.u_mul_e('h', 'feat', 'm'),
                        fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, norm)

        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class GcnSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_pp=False,
                 add_attn=False):
        super(GcnSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.add_attn = add_attn
        self.fcg = nn.Linear(4, 100)
        self.fcs = nn.Linear(300, 100)
        self.fcr = nn.Linear(512, 100)
        self.actt = nn.LeakyReLU()
        self.norm = nn.LayerNorm(100)

        # input layer
        self.layers.append(GcnSAGELayer(in_feats, n_hidden, activation=activation,
                                        dropout=dropout, use_pp=use_pp, use_lynorm=True))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GcnSAGELayer(n_hidden, n_hidden, activation=activation, dropout=dropout,
                            use_pp=False, use_lynorm=True))
        # output layer
        if not self.add_attn:
            self.layers.append(GcnSAGELayer(n_hidden, n_classes, activation=None,
                                        dropout=False, use_pp=False, use_lynorm=False))
        else:
            self.layers.append(GATConv(n_hidden, n_classes, num_heads=1))

    def forward(self, g, feat): #, padding=False):
        # TODO add FCN
        # x = feat[:n] where n is the length of geometric features
        # x = self.fc(x)
        # h = torch.cat(x, feat[n:])
        # print(feat.shape)
        fg = self.fcg(feat[:,:4])
        fg = self.norm(fg)
        fg = self.actt(fg)
        # print(fg.shape)

        fs = self.fcs(feat[:,4:])
        fs = self.norm(fs)
        fs = self.actt(fs)
        # print(fs.shape)

        h = torch.cat((fg, fs), 1)
        # print(h.shape)
                
        # h = self.dropout(h)
        for l in range(self.n_layers - 1):
            h = self.layers[l](g, h)
        
        if self.add_attn:
            h, a = self.layers[-1](g, h, True)
            return h.mean(1), a
        else:
            h = self.layers[-1](g, h)
            return h, None

#TODO
#! implement this method to train end-to-end
# class Visual_GcnSAGE(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout,
#                  use_pp=False,
#                  add_attn=False):
#         super(GcnSAGE, self).__init__()
#         self.backbone = torchvision.models.resnet50(pretrained=False)
#         self.layers = nn.ModuleList()
#         self.dropout = nn.Dropout(dropout)
#         self.n_layers = n_layers
#         self.add_attn = add_attn

#         self.backbone = torch.nn.Sequential(*self.backbone.children[:-2])

#         # input layer
#         self.layers.append(GcnSAGELayer(in_feats, n_hidden, activation=activation,
#                                         dropout=dropout, use_pp=use_pp, use_lynorm=True))
#         # hidden layers
#         for i in range(1, n_layers - 1):
#             self.layers.append(
#                 GcnSAGELayer(n_hidden, n_hidden, activation=activation, dropout=dropout,
#                             use_pp=False, use_lynorm=True))
#         # output layer
#         if not self.add_attn:
#             self.layers.append(GcnSAGELayer(n_hidden, n_classes, activation=None,
#                                         dropout=False, use_pp=False, use_lynorm=False))
#         else:
#             self.layers.append(GATConv(n_hidden, n_classes, num_heads=1))

#     def forward(self, img, bboxs, g): #, padding=False):
#         # TODO
#         # https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html?highlight=roi
#         # # rescale img to fixed
#         # visual_emb = self.backbone(img) # output [batch, canali, dim1, dim2]
#         # h = torchvision.ops.roi_align(visual_emb, bboxes, scaling(image->map))
                
#         # h = self.dropout(h)
#         for l in range(self.n_layers - 1):
#             h = self.layers[l](g, h)
        
#         if self.add_attn:
#             h, a = self.layers[-1](g, h, True)
#             return h.mean(1), a
#         else:
#             h = self.layers[-1](g, h)
#             return h, None
