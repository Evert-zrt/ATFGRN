import csv
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, CitationFull, Flickr, Twitch, Coauthor
from torch_geometric.utils import from_networkx, train_test_split_edges, add_self_loops, negative_sampling, k_hop_subgraph
from torch_geometric.transforms import RandomLinkSplit
import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os.path as osp
from itertools import chain

from .func_utils import extract_enclosing_subgraphs, drnl_node_labeling


class Gene(InMemoryDataset):
    def __init__(self, root, data_name,transform=None, pre_transform=None):
        self.data_name = data_name
        super().__init__(root, transform, pre_transform)

        self.data = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return ['gene.pt']

    def process(self):
        # data_dir = '../data/hESC 500'
        expression_data_path = self.data_name
        # edges = pd.read_csv(data_dir + '/Label.csv', index_col=0, header=0)
        expression_data = np.array(pd.read_csv(expression_data_path, index_col=0, header=0))
        # Data Preprocessing
        standard = StandardScaler()
        scaled_df = standard.fit_transform(expression_data.T)
        expression_data = scaled_df.T
        dataset = Data(
            x=torch.tensor(expression_data, dtype=torch.float),
            # edge_index=torch.tensor(edges.values).t().contiguous(),
            # y=torch.tensor(label)
        )
        torch.save(dataset, self.processed_paths[0])
def load_data(data_name,save_path):

    dataset = Gene(root=save_path, data_name=data_name)

    return dataset


def sample_dataset(edge_index, ratio):
    if ratio != 1.0:
        num_edges = edge_index.size(1)
        edge_index = edge_index[:, np.random.permutation(num_edges)[: int(ratio * num_edges)]]
    return edge_index


class ATFGRN_Dataset(InMemoryDataset):
    def __init__(self, dataset, args, num_hops=1, split='train'):
        self.dataset = dataset
        self.data_name = str(dataset)[:-2]
        self.data = dataset[0]
        self.args = args
        self.num_hops = num_hops
        super(ATFGRN_Dataset, self).__init__(dataset.root)
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])
    @property
    def processed_file_names(self):
        return ['{}_train_data.pt'.format(self.data_name),
                '{}_val_data.pt'.format(self.data_name),
                '{}_test_data.pt'.format(self.data_name)]


    def load_csv_data(self, file_path):

        df = pd.read_csv(file_path, index_col=0, header=0)

        pos_edges = df[df["Label"] == 1][["TF", "Target"]].values.tolist()
        neg_edges = df[df["Label"] == 0][["TF", "Target"]].values.tolist()

        pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
        return pos_edge_index, neg_edge_index

    def process(self):

        data_dir = '../data/' + self.args.netType + '/' + self.args.dataset + ' ' + self.args.num
        train_file = data_dir + '/Train_set.csv'
        val_file = data_dir + '/Validation_set.csv'
        test_file = data_dir + '/Test_set.csv'

        train_pos_edge_index, train_neg_edge_index = self.load_csv_data(train_file)

        val_pos_edge_index, val_neg_edge_index = self.load_csv_data(val_file)

        test_pos_edge_index, test_neg_edge_index = self.load_csv_data(test_file)

        data = self.data

        data.train_pos_edge_index = train_pos_edge_index
        data.train_neg_edge_index = train_neg_edge_index
        data.val_pos_edge_index = val_pos_edge_index
        data.val_neg_edge_index = val_neg_edge_index
        data.test_pos_edge_index = test_pos_edge_index
        data.test_neg_edge_index = test_neg_edge_index

        edge_index = data.train_pos_edge_index
        train_pos_edge_index = data.train_pos_edge_index
        train_neg_edge_index = data.train_neg_edge_index
        val_pos_edge_index = data.val_pos_edge_index
        val_neg_edge_index = data.val_neg_edge_index
        test_pos_edge_index = data.test_pos_edge_index
        test_neg_edge_index = data.test_neg_edge_index



        train_pos_list = extract_enclosing_subgraphs(self.data, self.num_hops, train_pos_edge_index, edge_index, 1, node_label='drnl')
        train_neg_list = extract_enclosing_subgraphs(self.data, self.num_hops, train_neg_edge_index, edge_index, 0, node_label='drnl')
        val_pos_list = extract_enclosing_subgraphs(self.data, self.num_hops, val_pos_edge_index, edge_index, 1, node_label='drnl')
        val_neg_list = extract_enclosing_subgraphs(self.data, self.num_hops, val_neg_edge_index, edge_index, 0, node_label='drnl')
        test_pos_list = extract_enclosing_subgraphs(self.data, self.num_hops, test_pos_edge_index, edge_index, 1, node_label='drnl')
        test_neg_list = extract_enclosing_subgraphs(self.data, self.num_hops, test_neg_edge_index, edge_index, 0, node_label='drnl')

        max_z = 0
        for data in chain(train_pos_list, train_neg_list, val_pos_list, val_neg_list, test_pos_list, test_neg_list):
            max_z = max(int(data.z.max()), max_z)
        for data in chain(train_pos_list, train_neg_list, val_pos_list, val_neg_list, test_pos_list, test_neg_list):
            data.x = F.one_hot(data.z, max_z + 1).to(torch.float)
            data.z = None

        torch.save(self.collate(train_pos_list + train_neg_list), self.processed_paths[0])
        torch.save(self.collate(val_pos_list + val_neg_list), self.processed_paths[1])
        torch.save(self.collate(test_pos_list + test_neg_list), self.processed_paths[2])


