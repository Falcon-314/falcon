# ====================
# cvsplit
# ====================

# ====================
# module import
# ====================
import random
import pandas as pd
import numpy as np

from collections import Counter, defaultdict

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

# =====================
# define function
# =====================
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

# =====================
# CVsplit
# =====================
def cvsplit(CFG, train):
    if CFG.fold_type == 'KFold':
        Fold = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train,train[CFG.target_cols])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)
        
    elif CFG.fold_type == 'StratifiedKFold':
        Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.stratified_col])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)

    elif CFG.fold_type == 'GroupKFold':
        Fold = GroupKFold(n_splits=CFG.n_fold)
        for n, (train_index, val_index) in enumerate(Fold.split(train,train[CFG.target_cols],train[CFG.group_col])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)
        
    elif CFG.fold_type == 'StratifiedGroupKFold':
        groups = np.array(train[CFG.group_col].values)
        train_y = train[CFG.target_cols].values.reshape(-1)
        for n, (train_index, val_index) in enumerate(stratified_group_k_fold(train, train_y, groups , CFG.n_fold)):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)
    
    else:
        print('fail to make fold')
        
 


