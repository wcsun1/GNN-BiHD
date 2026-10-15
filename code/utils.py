import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import KFold

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_sparse(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def reconstruct(pi_dis):
    score = []
    pi = []
    dis = []
    for i in range(len(pi_dis)):
        for j in range(pi_dis.shape[1]):
            score.append(pi_dis.iloc[i, j])
            pi.append(i)
            dis.append(j + len(pi_dis))
    return pd.DataFrame({0: pi, 1: dis, 2: score})


def symmetric_normalize(adjacency):
    degree = np.array(adjacency.sum(1)).flatten()
    scale = np.power(degree, -0.5)
    scale[np.isinf(scale)] = 0.0
    return sp.diags(scale).dot(adjacency).dot(sp.diags(scale))


def load_data(dataset, repeat, device, seed=42):
    path = './dataset/{}/'.format(dataset)
    features = pd.read_csv(path + '{}_feature.csv'.format(dataset), header=None).values
    association = pd.read_csv(path + '{}_AssociationMatrix.csv'.format(dataset), header=None)

    edge = reconstruct(association)
    edge_true = edge.loc[edge[2] == 1].reset_index(drop=True)
    edge_false = edge.loc[edge[2] == 0].reset_index(drop=True)
    rng = np.random.RandomState(seed)
    negative_index = rng.choice(len(edge_false), size=len(edge_true), replace=False)
    edge = pd.concat([edge_true, edge_false.iloc[negative_index]], ignore_index=True)
    edge = edge.iloc[rng.permutation(len(edge))].reset_index(drop=True)

    kfold = KFold(n_splits=5, shuffle=False, random_state=None)
    train_splits = []
    test_splits = []
    for train_index, test_index in kfold.split(edge[[0, 1]]):
        train_splits.append(train_index)
        test_splits.append(test_index)
    train_edge = edge.iloc[train_splits[repeat]].reset_index(drop=True)
    test_edge = edge.iloc[test_splits[repeat]].reset_index(drop=True)
    train_positive = train_edge.loc[train_edge[2] == 1]

    node_count = features.shape[0]
    mirna = train_positive[0].to_numpy(dtype=np.int64)
    disease = train_positive[1].to_numpy(dtype=np.int64)
    loops = np.arange(node_count, dtype=np.int64)
    sources = np.concatenate([mirna, disease, loops])
    targets = np.concatenate([disease, mirna, loops])
    pairs = np.unique(np.stack([sources, targets], axis=1), axis=0)
    binary = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(node_count, node_count))
    normalized = symmetric_normalize(binary.tocsr()).tocoo()
    adjacency = torch.sparse_coo_tensor(
        np.vstack([normalized.row, normalized.col]),
        normalized.data.astype(np.float32),
        (node_count, node_count),
    ).to_dense().to(device)

    features = torch.FloatTensor(np.nan_to_num(features.astype(np.float32))).to(device)
    edge_index = torch.tensor(pairs.T, dtype=torch.long, device=device)
    train_src = torch.tensor(train_edge[0].to_numpy(), dtype=torch.long, device=device)
    train_dst = torch.tensor(train_edge[1].to_numpy(), dtype=torch.long, device=device)
    train_y = torch.tensor(train_edge[2].to_numpy(), dtype=torch.float32, device=device)
    test_src = torch.tensor(test_edge[0].to_numpy(), dtype=torch.long, device=device)
    test_dst = torch.tensor(test_edge[1].to_numpy(), dtype=torch.long, device=device)
    test_y = torch.tensor(test_edge[2].to_numpy(), dtype=torch.float32, device=device)
    return {
        "features": features,
        "adj": adjacency,
        "edge_index": edge_index,
        "train_src": train_src,
        "train_dst": train_dst,
        "train_y": train_y,
        "test_src": test_src,
        "test_dst": test_dst,
        "test_y": test_y,
    }


def auroc(prob, label):
    y_true = label.data.cpu().numpy().flatten()
    y_scores = prob.data.cpu().numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score, fpr, tpr


def auprc(prob, label):
    y_true = label.data.cpu().numpy().flatten()
    y_scores = prob.data.cpu().numpy().flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc_score = auc(recall, precision)
    return auprc_score, precision, recall


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x


def prediction(predlabel, labels):
    predlabel_s = []
    labels_s = []
    for x in flatten(predlabel):
        predlabel_s.append(x)
    for x in flatten(labels):
        labels_s.append(x)
    acc = accuracy_score(labels_s, predlabel_s)
    precision = precision_score(labels_s, predlabel_s)
    recall = recall_score(labels_s, predlabel_s)
    f1 = f1_score(labels_s, predlabel_s)
    return acc, precision, recall, f1
