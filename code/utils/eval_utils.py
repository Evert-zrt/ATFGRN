import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_auc_ap(y_pred, y_true):
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise ValueError('Both y_pred and y_true need to be torch.Tensor.')
    y_pred = torch.sigmoid(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    result = {'AUC': auc, 'AP': ap}
    return result
