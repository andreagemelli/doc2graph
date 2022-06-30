from turtle import forward
from numpy import True_
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch.conv import GATv2Conv, SAGEConv
import math
import torch.nn.functional as F

from src.paths import MODELS
from src.amazing_utils import get_config

class SetModel():
    def __init__(self, name='gcn', device = 'cpu') -> None:

        self.cfg_model = get_config(MODELS / name)
        self.name = self.cfg_model.name
        self.total_params = 0
        self.device = device
    
    def get_name(self):
        return self.name
    
    def get_total_params(self):
        return self.total_params

    def get_model(self, nodes, edges, chunks):
        """Return the DGL model defined in the setting file

        Args:
            nums (list): num_features and num_classes

        Returns:
            A PyTorch Model
        """
        print("\n### MODEL ###")
        print(f"-> Using {self.name}")

        if self.name == 'GCN':
            m = GcnSAGE(chunks, self.cfg_model.out_chunks, nodes, self.cfg_model.num_layers, F.relu, self.cfg_model.dropout, self.cfg_model.attn)
        
        elif self.name == 'EDGE':
            m = EdgeClassifier(edges, self.cfg_model.num_layers, self.cfg_model.dropout, chunks, self.cfg_model.out_chunks, self.cfg_model.hidden_dim, self.device, self.cfg_model.doProject)

        elif self.name == 'GAT':
            m = EdgeClassifierGAT(nodes, edges, self.cfg_model.num_layers, chunks, self.cfg_model.out_chunks, self.device)

        else:
            raise Exception(f"Error! Model {self.name} do not exists.")
        
        m.to(self.device)
        self.total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"-> Total params: {self.total_params}")
        print("-> Device: " + str(next(m.parameters()).is_cuda) + "\n")
        print(m)
        return m

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
            # norm = self.get_norm(g)
            norm = 1
            #TODO concatenate edge reciprocal position information
            g.ndata['h'] = h
            g.update_all(fn.u_mul_e('h', 'weights', 'm'),
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
                 in_chunks,
                 out_chunks,
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

        self.projector = InputProjector(in_chunks, out_chunks)
        n_hidden = int(self.projector.get_out_lenght() / 2)

        # input layer
        self.layers.append(GcnSAGELayer(self.projector.get_out_lenght(), n_hidden, activation=activation,
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
            self.layers.append(GATv2Conv(n_hidden, n_classes, num_heads=1))

    def forward(self, g, h): #, padding=False):
        
        h = self.projector(h)

        for l in range(self.n_layers - 1):
            h = self.layers[l](g, h)
        
        if self.add_attn:
            h, a = self.layers[-1](g, h, True)
            return h.mean(1), a
        else:
            h = self.layers[-1](g, h)
            return h, None

class EdgeClassifier(nn.Module):

    def __init__(self, edge_classes, m_layers, dropout, in_chunks, out_chunks, hidden_dim, device, doProject=True):
        super().__init__()

        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        self.message_passing = nn.ModuleList()
        self.m_layers = m_layers
        for l in range(m_layers):
            self.message_passing.append(SAGEConv(m_hidden, m_hidden, 'mean', norm=nn.LayerNorm(m_hidden), activation=F.relu))

        # Define edge predictori layer
        self.edge_pred = MLPPredictor(m_hidden, hidden_dim, edge_classes, dropout)  

    def forward(self, g, h):

        h = self.projector(h)

        for l in range(self.m_layers):
            h = self.message_passing[l](g, h, g.edata['weights'])
        
        e = self.edge_pred(g, h)

        return e

class InputProjector(nn.Module):
    def __init__(self, in_chunks : list, out_chunks : int, device, doIt = True) -> None:
        super().__init__()
        
        if not doIt:
            self.output_length = sum(in_chunks)
            self.doIt = doIt
            return

        self.output_length = len(in_chunks)*out_chunks
        self.doIt = doIt
        self.chunks = in_chunks
        modules = []
        self.device = device

        for chunk in in_chunks:
            chunk_module = []
            chunk_module.append(nn.Linear(chunk, out_chunks))
            chunk_module.append(nn.LayerNorm(out_chunks))
            chunk_module.append(nn.ReLU())
            modules.append(nn.Sequential(*chunk_module))
        
        self.modalities = nn.Sequential(*modules)
        self.chunks.insert(0, 0)
    
    def get_out_lenght(self):
        return self.output_length
    
    def forward(self, h):

        if not self.doIt:
            return h

        out = []

        for name, module in self.modalities.named_children():
            num = int(name)
            if num + 1 == len(self.chunks): break
            start = self.chunks[num] + sum(self.chunks[:num])
            end = start + self.chunks[num+1]
            input = h[:, start:end].to(self.device)
            out.append(module(input))

        return torch.cat(out, dim=1)

class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features*2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim + 2, out_classes)
        self.drop = nn.Dropout(dropout)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        polar = edges.data['feat']

        try:
            e = edges.data['att']
            x = F.relu(self.norm(self.W1(e * (h_u + h_v))))
            score = self.drop(self.W2(x))
        except:
            x = F.relu(self.norm(self.W1(torch.cat((h_u, h_v), dim=1))))
            x = torch.cat((x, polar), dim=1)
            score = self.drop(self.W2(x))

        if self.out == 1:
            return {'score': score.squeeze(1)}
        else:
            return {'score': score}

    def forward(self, graph, h, att = None):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            if att is not None: graph.edata['att'] = att
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class EdgeClassifierGAT(nn.Module):
    def __init__(self, node_classes, edge_classes, m_layers, in_chunks, out_chunks, device, doProject = True):
        super().__init__()
        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        self.message_passing = nn.ModuleList()
        self.message_passing.append(GATv2Conv(self.projector.get_out_lenght(), m_hidden, num_heads=1, allow_zero_in_degree = True))
        self.m_layers = m_layers
        for l in range(m_layers - 1):
            self.message_passing.append(GATv2Conv(m_hidden, m_hidden, num_heads=1, allow_zero_in_degree = True))

        # Define edge predictori layer
        self.edge_pred = MLPPredictor(m_hidden, edge_classes)
        
        # Evaluate also node classes
        self.node_classes = node_classes
        if node_classes:
            self.node_pred = GATv2Conv(m_hidden, node_classes, num_heads=3)

    def forward(self, g, h):

        h = self.projector(h)

        for l in range(self.m_layers):
            h, att = self.message_passing[l](g, h, get_attention = True)

        e = self.edge_pred(g, h, att)

        if self.node_classes:
            return self.node_pred[-1](g, h), e
        else:
            return e