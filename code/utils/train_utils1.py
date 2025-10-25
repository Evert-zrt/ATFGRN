import os
import random

import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import KNNGraph
# from models.constrastive import Contrastive_Net
from torch_geometric.data import Data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#seed_all(2)
def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    for param in config_dict:
        default, type, description = config_dict[param]
        parser.add_argument(f"--{param}", default=default, type=type, help=description)

    return parser


def construct_knn_graph(data):
    if (not data.pos) and (data.x is not None):
        data.pos = data.x
    else:
        raise ValueError('No data pos and data features!')
    k = 1
    trans = KNNGraph(k, loop=False, force_undirected=True,cosine=True)
    knn_graph = trans(data.clone().to(device)).to(device).to('cpu')
    print("k=",knn_graph.edge_index.shape[1])

    data.pos, knn_graph.pos, knn_graph.x = None, None, None
    return knn_graph


def train_node2vec_emb(data):
    print('=' * 50)
    print('Start train node2vec model on the knn graph.')
    model = Node2Vec(data.edge_index, embedding_dim=32, walk_length=10, context_size=5, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=False, num_nodes=data.num_nodes)
    loader = model.loader(batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    minimal_loss = 1e9
    patience = 0
    patience_threshold = 20
    for epoch in range(1, 200):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
        if loss < minimal_loss:
            minimal_loss = loss
            patience = 0
        else:
            patience += 1
        if patience >= patience_threshold:
            print('Early Stop.')
            break
        print("Epoch: {:02d}, loss: {:.4f}".format(epoch, loss))
    print('Finished training.')
    print('=' * 50)
    return model().detach()


from torch_geometric.nn import GCNConv, GAE
import torch
import torch.nn.functional as F

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train_gae_embedding(data, embedding_dim=32, epochs=200, lr=0.01):
    print('=' * 50)
    print('Start training GAE to generate structural node features.')

    x = torch.eye(data.num_nodes).to(data.edge_index.device)


    encoder = GCNEncoder(in_channels=data.num_nodes, out_channels=embedding_dim)
    model = GAE(encoder).to(data.edge_index.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    print('Finished training GAE.')
    print('=' * 50)

    return model.encode(x, data.edge_index).detach()