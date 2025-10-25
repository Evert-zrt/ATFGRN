import math
import os
import random
from itertools import combinations, product
from torch_geometric.utils import to_undirected
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Conv1d, MaxPool1d, Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GCNConv, global_sort_pool, GATConv, MessagePassing, GATv2Conv, global_mean_pool, \
    JumpingKnowledge, TransformerConv
from torch_geometric.nn import GCN, GAT
from torch.nn import MultiheadAttention
from torch_geometric.utils import add_remaining_self_loops
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PatchySANPooling(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, batch):

        pooled = []
        for i in batch.unique():
            idx = (batch == i).nonzero(as_tuple=True)[0]
            x_sub = x[idx]
            degrees = x_sub.norm(p=2, dim=1)
            topk = torch.topk(degrees, k=min(self.k, x_sub.size(0)), largest=True).indices
            patch = x_sub[topk]
            if patch.size(0) < self.k:
                pad = torch.zeros(self.k - patch.size(0), x.size(1), device=x.device)
                patch = torch.cat([patch, pad], dim=0)
            pooled.append(patch.view(-1))
        return torch.stack(pooled)



class SubgraphContrastiveEncoder(nn.Module):
    def __init__(self,train_dataset, in_channels, hidden_channels, out_channels, num_layers,k):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.pool = PatchySANPooling(self.k)  # new!
        self.proj_head = nn.Sequential(
            Linear(hidden_channels * k, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels)
        )
    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.pool(x, batch)
        return self.proj_head(x)
    def forward(self, data):
        z1 = self.encode(data.x, data.edge_index, data.batch)
        return z1

class MultiScaleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        self.jk = JumpingKnowledge(mode='lstm', channels=hidden_channels, num_layers=num_layers)
        self.out_proj = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)
        x = self.jk(xs)  # [N, hidden]
        return self.out_proj(x)



class GRNTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, heads=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.linear(x))
        h1 = self.conv1(x, edge_index)
        x = self.norm1(h1 + x)
        h2 = self.conv2(x, edge_index)
        x = self.norm2(h2 + x)
        return x
class GATS(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super(GATS, self).__init__()
        self.dropout = dropout

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.gat2 = GATConv(hidden_channels, out_channels, heads=heads, concat=False)
        self.norm2 = nn.LayerNorm(out_channels)


        if in_channels != hidden_channels:
            self.res_fc1 = nn.Linear(in_channels, hidden_channels)
        else:
            self.res_fc1 = nn.Identity()

        if hidden_channels != out_channels:
            self.res_fc2 = nn.Linear(hidden_channels, out_channels)
        else:
            self.res_fc2 = nn.Identity()

    def forward(self, x, edge_index):

        x_input = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.gat1(x, edge_index)
        x1 = self.norm1(x1)
        x = F.elu(x1 + self.res_fc1(x_input))


        x_input2 = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x2 = self.gat2(x, edge_index)
        x2 = self.norm2(x2)
        x = x2 + self.res_fc2(x_input2)

        return x
class ATFGRN(torch.nn.Module):
    def __init__(self, train_dataset, grn_size, in_channels, hidden_channels, out_channels, num_layers,contrastive_margin=1.0):
        super(ATFGRN, self).__init__()
        self.gclencoder = SubgraphContrastiveEncoder(train_dataset,in_channels, hidden_channels, out_channels, num_layers=num_layers,k=0.2)
        self.Mgcn = MultiScaleGNN(in_channels=32, hidden_channels=hidden_channels, out_channels=out_channels,
                                  num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(grn_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),

        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )


        # self.gcn = GATS(in_channels=grn_size, hidden_channels=128, out_channels=128)
        self.gcn = GRNTransformer(input_dim=grn_size, hidden_dim=128, heads=8)
        # self.gcn = GRNSAGE(input_dim=grn_size, hidden_dim=128)
        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin_g = Linear(out_channels, 1)
        self.lin_grn = Linear(128*2, 32)
        self.lin3 = Linear(out_channels, 1)
        self.lin4  = nn.Sequential(

            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),

        )
        self.fuse_mlp  = nn.Sequential(
            nn.Linear(out_channels*3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.att_grn = Linear(out_channels, 16)
        self.query = Linear(16, 1)

    def undirected(self,edge_index):

        reversed_edge = edge_index.flip(0)
        edge_index_undirected = torch.cat([edge_index, reversed_edge], dim=1)

        edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
        return edge_index_undirected


    def forward(self, data,grn_data, knn_graph, compute_contrastive=False):
        emb_2 = self.Mgcn(x=knn_graph.x, edge_index=knn_graph.edge_index)
        # emb_grn = self.mlp(grn_data.x) # 32



        grn_data.edge_index = self.undirected(grn_data.edge_index)

        emb_grn = self.gcn(x=grn_data.x, edge_index=grn_data.edge_index) # 32
        emb_grn = self.lin4(emb_grn)
        emb_grn = emb_grn[data.target_nodes]

        emb_2 = emb_2[data.target_nodes]
        emb_1= self.gclencoder(data)

        N = emb_2.size(0)
        index_1 = torch.range(0, N-1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        emb_2 = emb_2[index_1] + emb_2[index_2]
        emb_grn = emb_grn[index_1] + emb_grn[index_2]


        #1
        att_1 = self.query(F.tanh(self.att_1(emb_1)))
        att_2 = self.query(F.tanh(self.att_2(emb_2)))
        att_grn = self.query(F.tanh(self.att_grn(emb_grn)))
        alpha_t = torch.exp(att_1) / (torch.exp(att_1) + torch.exp(att_2) + torch.exp(att_grn))
        alpha_f = torch.exp(att_2) / (torch.exp(att_1) +torch.exp(att_2) + torch.exp(att_grn))
        alpha_g = torch.exp(att_grn) / ( torch.exp(att_1) +torch.exp(att_2) + torch.exp(att_grn))
        x = torch.cat([alpha_t*emb_1,alpha_f*emb_2,alpha_g * emb_grn], dim=-1)


        output_1 = self.lin1(emb_1)
        output_2 = self.lin2(emb_2)
        output_g = self.lin_g(emb_grn)
        output_3 = self.fuse_mlp(x)

        return output_g,output_1, output_2, output_3
