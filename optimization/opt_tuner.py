import optuna.integration.lightgbm as lgb_tuner
import lightgbm as lgb

def lgbtuner(folds, fold, param, features, target_col, get_score):

    # ====================================================
    # dataset
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    trainData = lgb.Dataset(train_folds[features],train_folds[target_col])
    validData = lgb.Dataset(valid_folds[features],valid_folds[target_col])

    # ====================================================
    # train
    # ====================================================
       
    # train
    model = lgb_tuner.train(param,
                  trainData,
                  valid_sets = [trainData, validData],
                  num_boost_round = 10000,
                  early_stopping_rounds = 100,
                  verbose_eval = -1,
                  show_progress_bar = False)

    # eval
    y_pred_valid = model.predict(valid_folds[features])
            
    # scoring
    score = get_score(valid_folds[target_col], y_pred_valid)

    best_params = model.params

    return score, best_params
