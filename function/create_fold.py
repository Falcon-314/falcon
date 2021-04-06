#cvsplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

def cvsplit(train, CFG):
    if CFG.fold_type == 'KFold':
        Fold = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train,train[CFG.target_col])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)
        
    elif CFG.fold_type == 'StratifiedKFold':
        Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.stratified_col])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)

    elif CFG.fold_type == 'GroupKFold':
        Fold = GroupKFold(n_splits=CFG.n_fold)
        for n, (train_index, val_index) in enumerate(Fold.split(train,train[CFG.target_col],train[CFG.group_col])):
            train.loc[val_index, 'fold'] = int(n)
        train['fold'] = train['fold'].astype(int)
    
    else:
        print('fail to make fold')
