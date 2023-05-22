import torch
import torch.nn as nn
import dgl.function as fn
import math
import torch.nn.functional as F

from src.paths import CFGM
from src.utils import get_config

class SetModel():
    def __init__(self, name='e2e', device = 'cpu'):
        """ Create a SetModel object, that handles dinamically different version of Doc2Graph Model. Default "end-to-end" (e2e)

        Args:
            name (str) : Which model to train / test. Default: e2e [gcn, edge].
        
        Returns:
            SetModel object.
        """

        self.cfg_model = get_config(CFGM / name)
        self.name = self.cfg_model.name
        self.total_params = 0
        self.device = device
    
    def get_name(self) -> str:
        """ Returns model name.
        """
        return self.name
    
    def get_total_params(self) -> int:
        """ Returns number of model parameteres.
        """
        return self.total_params

    def get_model(self, nodes : int, edges : int, chunks : list, verbatim : bool = True) -> nn.Module:
        """Return the DGL model defined in the setting file

        Args:
            nodes (int) : number of nodes target class
            edges (int) : number of edges target class
            chunks (list) : list of indeces of chunks

        Returns:
            A PyTorch nn.Module, your DGL model.
        """
        print("\n### MODEL ###")
        print(f"-> Using {self.name}")

        if self.name == 'GCN':
            m = NodeClassifier(chunks, self.cfg_model.out_chunks, nodes, self.cfg_model.num_layers, F.relu, False, self.device)
        
        elif self.name == 'EDGE':
            m = EdgeClassifier(edges, self.cfg_model.num_layers, self.cfg_model.dropout, chunks, self.cfg_model.out_chunks, self.cfg_model.hidden_dim, self.device, self.cfg_model.doProject)

        elif self.name == 'E2E':
            edge_pred_features = int((math.log2(get_config('preprocessing').FEATURES.num_polar_bins) + nodes)*2)
            m = E2E(nodes, edges, self.cfg_model.num_layers, self.cfg_model.dropout, chunks, self.cfg_model.out_chunks, self.cfg_model.hidden_dim, self.device,  edge_pred_features, self.cfg_model.doProject)

        else:
            raise Exception(f"Error! Model {self.name} do not exists.")
        
        m.to(self.device)
        self.total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"-> Total params: {self.total_params}")
        print("-> Device: " + str(next(m.parameters()).is_cuda) + "\n")
        if verbatim: print(m)

        return m

################
##### GCNS #####

class NodeClassifier(nn.Module):
    def __init__(self,
                 in_chunks,
                 out_chunks,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0,
                 use_pp=False,
                 device='cuda:0'):
        super(NodeClassifier, self).__init__()

        self.projector = InputProjector(in_chunks, out_chunks, device)
        self.layers = nn.ModuleList()
        # self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        n_hidden = self.projector.get_out_lenght()

        # mp layers
        for i in range(0, n_layers - 1):
            self.layers.append(GcnSAGELayer(n_hidden, n_hidden, activation=activation, 
                        dropout=dropout, use_pp=False, use_lynorm=True))

        self.layers.append(GcnSAGELayer(n_hidden, n_classes, activation=None,
                                    dropout=False, use_pp=False, use_lynorm=False))

    def forward(self, g, h):
        
        h = self.projector(h)

        for l in range(self.n_layers):
            h = self.layers[l](g, h)
        
        return h

################
##### EDGE #####

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
            self.message_passing.append(GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.))

        # Define edge predictori layer
        self.edge_pred = MLPPredictor(m_hidden, hidden_dim, edge_classes, dropout)  

    def forward(self, g, h):

        h = self.projector(h)

        for l in range(self.m_layers):
            h = self.message_passing[l](g, h)
        
        e = self.edge_pred(g, h)

        return e

################
###### E2E #####

class E2E(nn.Module):
    def __init__(self, node_classes, 
                       edge_classes, 
                       m_layers, 
                       dropout, 
                       in_chunks, 
                       out_chunks, 
                       hidden_dim, 
                       device,
                       edge_pred_features,
                       doProject=True):

        super().__init__()

        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        self.message_passing = nn.ModuleList()
        # self.m_layers = m_layers
        # for l in range(m_layers):
        #     self.message_passing.append(GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.))
        self.message_passing = GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.)

        # Define edge predictor layer
        self.edge_pred = MLPPredictor_E2E(m_hidden, hidden_dim, edge_classes, dropout,  edge_pred_features)

        # Define node predictor layer
        node_pred = []
        node_pred.append(nn.Linear(m_hidden, node_classes))
        node_pred.append(nn.LayerNorm(node_classes))
        self.node_pred = nn.Sequential(*node_pred)

    def forward(self, g, h):

        h = self.projector(h)
        # for l in range(self.m_layers):
        #     h = self.message_passing[l](g, h)
        h = self.message_passing(g,h)
        n = self.node_pred(h)
        e = self.edge_pred(g, h, n)
        
        return n, e

################
##### LYRS #####

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
            norm = g.ndata['norm']
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

        mid = []

        for name, module in self.modalities.named_children():
            num = int(name)
            if num + 1 == len(self.chunks): break
            start = self.chunks[num] + sum(self.chunks[:num])
            end = start + self.chunks[num+1]
            input = h[:, start:end].to(self.device)
            mid.append(module(input))

        return torch.cat(mid, dim=1)

class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features*2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim + 6, out_classes)
        self.drop = nn.Dropout(dropout)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        polar = edges.data['feat']

        x = F.relu(self.norm(self.W1(torch.cat((h_u, h_v), dim=1))))
        x = torch.cat((x, polar), dim=1)
        score = self.drop(self.W2(x))

        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class MLPPredictor_E2E(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout,  edge_pred_features):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features*2 +  edge_pred_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_classes)
        self.drop = nn.Dropout(dropout)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        cls_u = F.softmax(edges.src['cls'], dim=1)
        cls_v = F.softmax(edges.dst['cls'], dim=1)
        polar = edges.data['feat']

        x = F.relu(self.norm(self.W1(torch.cat((h_u, cls_u, polar, h_v, cls_v), dim=1))))
        score = self.drop(self.W2(x))

        return {'score': score}

    def forward(self, graph, h, cls):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['cls'] = cls
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
