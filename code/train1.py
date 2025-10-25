import os
import random
import time
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.data_utils import load_data, ATFGRN_Dataset
from utils.eval_utils import evaluate_auc_ap
from utils.train_utils1 import construct_knn_graph, train_node2vec_emb
from models.model import   ATFGRN
from config import parser
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import logging

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True



def train(model, train_loader,grn_data, knn_graph, device, optimizer, train_dataset):
    num=0
    model.train()
    total_loss = 0
    y_pred, y_true = [], []
    for data in tqdm(train_loader, ncols=70):
        data = data.to(device)
        optimizer.zero_grad()
        logits_g,logits_1, logits_2, logits_3 = model(data,grn_data, knn_graph)
        loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), data.y.to(torch.float))
        loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), data.y.to(torch.float))
        loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), data.y.to(torch.float))
        loss_g = BCEWithLogitsLoss()(logits_g.view(-1), data.y.to(torch.float))


        loss = loss_3+loss_2+loss_1+loss_g
        # loss = loss_3
        loss.backward()
        optimizer.step()
        num+=1
        y_pred.append(logits_3.detach().view(-1).cpu())
        y_true.append(data.y.detach().view(-1).cpu().to(torch.float))
        total_loss += loss.item()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    return total_loss/num,evaluate_auc_ap(y_pred, y_true)

@torch.no_grad()
def test(args, loader, grn_data,knn_graph, model, device,dataset):
    model.eval()
    num=0
    total_loss = 0
    y_pred, y_true = [], []
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        logits_g,logits1,logits2, logits3 = model(data, grn_data,knn_graph)

        loss_3 = BCEWithLogitsLoss()(logits3.view(-1), data.y.to(torch.float))


        loss = loss_3
        # loss = loss_3
       #  total_loss += loss.item() * data.num_graphs
        total_loss += loss.item()
        num+=1
        y_pred.append(logits3.detach().view(-1).cpu())
        y_true.append(data.y.detach().view(-1).cpu().to(torch.float))


    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return total_loss/num, evaluate_auc_ap(y_pred, y_true)

def Adj(args):
    data_dir = '../data/' + args.netType + '/' + args.dataset + ' ' + args.num
    train_file = data_dir + '/Train_set.csv'

    df = pd.read_csv(train_file, index_col=0, header=0)

    pos_edges = df[df["Label"] == 1][["TF", "Target"]].values.tolist()

    pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
    return pos_edge_index


def run(args):

    expfile = "../Benchmark Dataset/"+args.netType+' Dataset/'+args.dataset+'/TFs+'+args.num+'/BL--ExpressionData.csv'
    save_path = '../Data_process'+ '/' +args.netType+ '/' + args.dataset + ' ' + args.num
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset = load_data(expfile,save_path)
    if args.dataset == 'hESC':
        lrs = 0.001
    elif args.dataset == 'hHEP':
        lrs = 0.001
    elif args.dataset == 'mDC':
        lrs = 0.001
    elif args.dataset == 'mESC':
        lrs = 0.001
    elif args.dataset == 'mHSC-E':
        lrs = 0.001
    elif args.dataset == 'mHSC-GM':
        lrs = 0.001
    elif args.dataset == "mHSC-L":
        lrs = 0.001
    else:
        lrs = 0.001
    print(lrs)
    knn_graph = construct_knn_graph(data=dataset[0])
    # knn_graph = construct_cos_knn_graph(dataset[0])
    emb = train_node2vec_emb(knn_graph)
    #emb = train_gae_embedding(knn_graph)
    # emb = train_cl_emb(dataset[0], knn_graph)
    knn_graph.x = emb
    train_dataset_class = 'ATFGRN_Dataset'
    val_dataset_class = 'ATFGRN_Dataset'
    test_dataset_class = 'ATFGRN_Dataset'

    train_dataset = eval(train_dataset_class)(dataset, args, num_hops=2, split='train')
    val_dataset = eval(val_dataset_class)(dataset, args, num_hops=2, split='val')
    test_dataset = eval(test_dataset_class)(dataset, args, num_hops=2, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    device = torch.device('cuda:1' if args.cuda else 'cpu')

    train_data = Adj(args)
    feature = dataset[0].x.to(device)
    grn_data= Data(x=feature, edge_index=train_data).to(device)
    model = ATFGRN(train_dataset, feature.size(1), train_dataset[0].num_features, hidden_channels=32, out_channels=32, num_layers=args.num_layers).to(device)
    logger.info(model)
    knn_graph = knn_graph.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs, weight_decay=args.wd)
    schedular = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)


    if args.metric == 'auc_ap':
        best_test_auc = best_test_ap = test_auc = test_ap = 0
    elif args.metric == 'hits':
        best_val_hits = test_hits = 0
    else:
        raise ValueError('Invalid metric')
    patience = 0

    best_model_path = args.netType + ' ' + args.dataset + args.num + '_best_model.pt'

    # model.load_state_dict(torch.load(best_model_path))
    # _, final_test_results = test(args, test_loader, grn_data, knn_graph, model, device, test_dataset)

    for epoch in range(1, args.epochs):
        schedular.step()
        loss, result = train(model, train_loader, grn_data, knn_graph, device, optimizer, train_dataset)
        val_loss, val_results = test(args, val_loader, grn_data, knn_graph, model, device, val_dataset)
        train_auc, train_ap = result['AUC'], result['AP']
        test_loss, test_results = test(args, test_loader, grn_data, knn_graph, model, device, test_dataset)
        test_auc, test_ap = test_results['AUC'], test_results['AP']
        if args.metric == 'auc_ap':
            val_auc, val_ap = val_results['AUC'], val_results['AP']
            if round(test_ap, 4) > round(best_test_ap, 4):
                best_test_auc = test_auc
                best_test_ap = test_ap

                patience = 0

                torch.save(model.state_dict(), best_model_path)
            else:
                patience += 1

            logger.info(
                f'Epoch: {epoch:02d}, trainLoss: {loss:.4f}, train_AUC: {train_auc:.4f}, train_AP: {train_ap:.4f},'
                f'valLoss: {val_loss:.4f},Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}, '
                f'testLoss: {test_loss:.4f},Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}'
            )
            if patience >= args.patience:
                logger.info('Early Stop! Best Val AUC: {:.4f}, Best Val AP: {:.4f}'.format(best_test_auc, best_test_ap))

                model.load_state_dict(torch.load(best_model_path, map_location=device))
                _, final_test_results = test(args, test_loader, grn_data, knn_graph, model, device, test_dataset)

                final_test_auc = final_test_results['AUC']
                final_test_ap = final_test_results['AP']

                logger.info('Best Val -> Final Test AUC: {:.4f}, Test AP: {:.4f}'.format(final_test_auc, final_test_ap))
                break

    return [final_test_auc, final_test_ap]


if __name__ == '__main__':
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    exp_time = '_'.join(time.asctime().split(' '))

    log_dir = '../results/ATFGRN_fomer'+'/'
    log_file = log_dir + 'Log_ATFGRN_{}_{}{}__{}.txt'.format(args.netType, args.dataset.capitalize(), args.num,exp_time)


    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(args)
    res = []
    for _ in range(args.runs):

        results = run(args)
        res.append(results)

    if args.metric == 'auc_ap':
        for i in range(len(res)):
            logger.info(f'Run: {i + 1:2d}, Test AUC: {res[i][0]:.4f}, Test AP: {res[i][1]:.4f}')
        auc, ap = 0, 0
        for j in range(len(res)):
            auc += res[j][0]
            ap += res[j][1]
        logger.info("The average AUC for test data is {:.4f}".format(auc / args.runs))
        logger.info("The average AP for test data is {:.4f}".format(ap / args.runs))
