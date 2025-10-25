import torch
import argparse
from utils.train_utils1 import add_flags_from_config

config_args = {
    'training_config': {
        'runs': (10, int, 'num of runs'),
        'lr': (0.001, float, 'learning rate'),
        'epochs': (401, int, 'training epochs'),
        'cuda': (torch.cuda.is_available(), bool, 'cuda'),
        'wd': (5e-4, float, 'weight decaying'),
        'bs': (32, int, 'batch size'),
        'patience': (10, int, 'early stop patience'),
        'num_layers':(2,int,'num of layers in GNN'),
        'ratio':(0.4,float,'ratio of training data'),
        'metric': ('auc_ap', str, 'performance metric to use'),
    },
    'data_config': {
        'netType':('Specific', str, 'which network to use'),
        'num': ('500', str, 'which network to use'),
        'dataset': ('mESC', str, 'which dataset to use'),
        'train_percent': (1.0, float, 'the ratio of links of the split edges'),
        'val_percent': (1.0, float, 'the ratio of links of the split edges'),
        'test_percent': (1.0, float, 'the ratio of links of the split edges'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
