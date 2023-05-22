import torch
import torch.nn as nn
import dgl.function as fn
import math
import torch.nn.functional as F
from dgl import DGLGraph
import torchvision
import dgl

from src.globals import DEVICE

################
###### E2E #####

class E2E(nn.Module):
    def __init__(self, node_classes, 
                       edge_classes,
                       in_chunks,
                       device,
                       edge_pred_features,
                       dropout, 
                       out_chunks, 
                       hidden_dim, 
                       doProject=True,
                       doPretrain=False):

        super().__init__()

        # visual embedder
        self.visual_embedder = VisualEmbedder()

        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        self.message_passing = nn.ModuleList()
        self.message_passing = GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.)

        if not doPretrain:
            # Define edge predictor layer
            self.edge_pred = MLPPredictor_E2E(m_hidden, hidden_dim, edge_classes, dropout,  edge_pred_features)

            # Define node predictor layer
            node_pred = []
            node_pred.append(nn.Linear(m_hidden, node_classes))
            node_pred.append(nn.LayerNorm(node_classes))
            self.node_pred = nn.Sequential(*node_pred)


    def forward(self, g, h, img=None, bboxs=None):
        
        if img is not None:
            x = self.visual_embedder(img, bboxs)
            h = torch.cat((h, x), dim=1)
        h = self.projector(h)
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
                 do_pretrain=False):
        super(GcnSAGELayer, self).__init__()
        self.linear = nn.Linear(5 * in_feats, out_feats)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        self.reset_parameters()
        self.do_pretrain = do_pretrain

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g : DGLGraph, h):
        g = g.local_var()
        g.ndata['h'] = h
        positions = g.edata['position']

        right = (positions == 0).nonzero(as_tuple=True)[0].tolist()
        g.send_and_recv(right, fn.u_mul_e('h', 'dist', 'm'), fn.sum(msg='m', out='r'))
        r = g.ndata.pop('r')

        bottom = (positions == 1).nonzero(as_tuple=True)[0].tolist()
        g.send_and_recv(bottom, fn.u_mul_e('h', 'dist', 'm'), fn.sum(msg='m', out='b'))
        b = g.ndata.pop('b')

        left = (positions == 2).nonzero(as_tuple=True)[0].tolist()
        g.send_and_recv(left, fn.u_mul_e('h', 'dist', 'm'), fn.sum(msg='m', out='l'))
        l = g.ndata.pop('l')

        top = (positions == 0).nonzero(as_tuple=True)[0].tolist()
        g.send_and_recv(top, fn.u_mul_e('h', 'dist', 'm'), fn.sum(msg='m', out='t'))
        t = g.ndata.pop('t')
        
        h = torch.cat((h, r, b, l, t), dim=1)

        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)

        if self.do_pretrain:
            with g.local_scope():
                g.ndata['h'] = h
                h = dgl.mean_nodes(g, 'h')

        return h

    # def forward(self, g : DGLGraph, h):
    #     g = g.local_var()
        
    #     if not self.use_pp:
    #         norm = g.ndata['norm']
    #         g.ndata['h'] = h
    #         g.update_all(fn.u_mul_e('h', 'dist', 'm'),
    #                     fn.sum(msg='m', out='h'))
    #         ah = g.ndata.pop('h')
    #         h = self.concat(h, ah, norm)

    #     if self.dropout:
    #         h = self.dropout(h)

    #     h = self.linear(h)
    #     h = self.lynorm(h)
    #     if self.activation:
    #         h = self.activation(h)
    #     return h

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
        # self.embedding = nn.Embedding(1000, 20)

        for chunk in in_chunks:
            chunk_module = []
            # if chunk == 6:
            #    chunk *= 20
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
            # if input.shape[1] == 6: 
            #     input = self.embedding(input.long())
            #     input = torch.reshape(input, (input.shape[0], -1))
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

class VisualEmbedder(nn.Module):
    def __init__(self):
        # first line
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.gn1 = nn.GroupNorm(32, 32)

        self.conv21 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3)
        self.gn2 = nn.GroupNorm(32, 64)

        # second line
        self.conv31 = nn.Conv2d(64, 128, kernel_size=(1,3), dilation=2)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=(3,1))
        self.conv33 = nn.Conv2d(128, 128, kernel_size=(1,3), dilation=4)
        self.conv34 = nn.Conv2d(128, 128, kernel_size=3)

        # third line
        self.conv41 = nn.Conv2d(128, 128, kernel_size=(1,3), dilation=4)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=(3,1))
        self.conv43 = nn.Conv2d(128, 128, kernel_size=(1,3), dilation=8)
        self.conv44 = nn.Conv2d(128, 128, kernel_size=3)
        self.gn34 = nn.GroupNorm(32, 128)

        # fourth line
        # self.conv51 = nn.Conv2d(128, 256, kernel_size=(1,3), dilation=8)
        # self.conv52 = nn.Conv2d(256, 256, kernel_size=(3,1))
        # self.conv53 = nn.Conv2d(256, 256, kernel_size=(1,3), dilation=4) # reduced from 16
        # self.conv54 = nn.Conv2d(256, 256, kernel_size=3)
        # self.conv55 = nn.Conv2d(256, 256, kernel_size=3)
        # self.gn5 = nn.GroupNorm(32, 256)

        # self.out = nn.Conv2d(256, 225, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.act = nn.ReLU()

    def forward(self, x, bboxs):

        h , w = x.shape[2], x.shape[3]

        x = self.act(self.gn1(self.conv1(x)))
        hook1 = self.pool(x)
        h1 = torchvision.ops.roi_align(input=hook1, boxes=bboxs, spatial_scale= min(hook1.shape[2] / h, hook1.shape[3] / w), output_size=4)
        h1 = torch.reshape(h1, (h1.shape[0], -1))

        x = self.act(self.gn2(self.conv21(hook1)))
        x = self.act(self.gn2(self.conv22(x)))
        x = self.pool(x)

        x = self.act(self.gn34(self.conv31(x)))
        x = self.act(self.gn34(self.conv32(x)))
        x = self.act(self.gn34(self.conv33(x)))
        x = self.act(self.gn34(self.conv34(x)))
        x = self.pool(x)

        x = self.act(self.gn34(self.conv41(x)))
        x = self.act(self.gn34(self.conv42(x)))
        x = self.act(self.gn34(self.conv43(x)))
        x = self.act(self.gn34(self.conv44(x)))
        hook2 = self.pool(x)
        h2 = torchvision.ops.roi_align(input=hook2, boxes=bboxs, spatial_scale=1 / min(h / hook2.shape[2] , w / hook2.shape[3]), output_size=2)
        h2 = torch.reshape(h2, (h2.shape[0], -1))

        # x = self.act(self.gn5(self.conv51(x)))
        # x = self.act(self.gn5(self.conv52(x)))
        # x = self.act(self.gn5(self.conv53(x)))
        # x = self.act(self.gn5(self.conv54(x)))
        # x = self.act(self.gn5(self.conv55(x)))
        # x = self.out(x)

        return torch.cat((h1, h2), dim=1)

# if __name__ == "__main__":
#     x = torch.rand(1, 1, 1000, 775)
#     ve = VisualEmbedder()
#     h1, h2 = ve(x)