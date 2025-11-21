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
    """
       Constructs a K-Nearest Neighbor (KNN) graph based on node feature similarity.
    """
    if (not data.pos) and (data.x is not None):
        data.pos = data.x
    else:
        raise ValueError('No data pos and data features!')

    # Initialize KNN (k=1, cosine similarity, undirected)
    k = 1
    trans = KNNGraph(k, loop=False, force_undirected=True,cosine=True)
    knn_graph = trans(data.clone().to(device)).to(device).to('cpu')
    print("k=",knn_graph.edge_index.shape[1])

    data.pos, knn_graph.pos, knn_graph.x = None, None, None
    return knn_graph


def train_node2vec_emb(data):
    """
    Trains a Node2Vec model to generate structural node embeddings.
    """
    print('=' * 50)
    print('Start train node2vec model on the knn graph.')

    # Initialize Node2Vec with random walk parameters
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
            loss = model.loss(pos_rw, neg_rw)# Contrastive loss
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

    # Return the learned node embeddings (detach from computation graph)
    return model().detach()
