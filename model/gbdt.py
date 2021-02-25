import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
from catboost import Pool

import wandb

from abc import abstractmethod
class Base_Model(object):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError
   

class Lgbm(Base_Model):
    def __init__(self,target_col,model_params,wandb,get_score,OUTPUT_DIR):
        self.model_params = model_params
        self.models = []

    def fit(self,x_train,y_train,x_valid,y_valid):
        lgb_train = lgb.Dataset(x_train,y_train)
        lgb_valid = lgb.Dataset(x_valid,y_valid)

        model = lgb.train(self.model_params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            categorical_feature=cat_col,
            early_stopping_rounds=20,
            num_boost_round=10000,
            verbose_eval=False)
        self.models.append(model)
        return model

    def predict(self,model,features):
        self.feature_cols = features.columns
        return model.predict(features)

    def visualize_importance(self):
        feature_importance_df = pd.DataFrame()

        for i,model in enumerate(self.models):
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importance(importance_type='gain')
            _df['column'] = self.feature_cols
            _df['fold'] = i+1
            feature_importance_df = pd.concat([feature_importance_df,_df],axis=0,ignore_index=True)

        order = feature_importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',ascending=False).index[:50]

        fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        fig.tight_layout()
        return fig,ax
 
from wandb.lightgbm import wandb_callback
    callbacks=[wandb_callback()]
    
LOGGER.info(f"========== fold: {fold} training ==========")

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
   
    start_time = time.time()
    
    # train
    model = lgb.train(param,
                  trainData,
                  valid_sets = [trainData, validData],
                  num_boost_round = 10000,
                  early_stopping_rounds = 100,
                  verbose_eval = -1)

    # eval
    y_pred_valid = model.predict(valid_folds[features])
            
    # scoring
    score = get_score(valid_folds[target_col], y_pred_valid)

    elapsed = time.time() - start_time

    LOGGER.info(f'Score: {score} - time: {elapsed:.0f}s')

    # modelのsave
    pickle.dump(model, open(OUTPUT_DIR+f'lgbm_fold{fold}.sav','wb'))
    
    # 出力用データセットへの代入
    valid_folds['preds'] = y_pred_valid
    
    #重要度の出力
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = fold

class Cat(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
    def fit(self,x_train,y_train,x_valid,y_valid):
        train_pool = Pool(x_train,
                          label=y_train,
                          cat_features=cat_col)
        valid_pool = Pool(x_valid,
                          label=y_valid,
                          cat_features=cat_col)

        model = CatBoost(self.model_params)
        model.fit(train_pool,
                  early_stopping_rounds=30,
                 plot=False,
                 use_best_model=True,
                 eval_set=[valid_pool],
                  verbose=False)

        return model

    def predict(self,model,features):
      pred = model.predict(features)
      return pred

class Xgb(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params

    def fit(self,x_train,y_train,x_valid,y_valid):
        xgb_train = xgb.DMatrix(x_train,label=y_train)
        xgb_valid = xgb.DMatrix(x_valid,label=y_valid)

        evals = [(xgb_train,'train'),(xgb_valid,'eval')]

        model = xgb.train(self.model_params,
                         xgb_train,
                         num_boost_round=2000,
                         early_stopping_rounds=20,
                         evals=evals,
                         verbose_eval=False)

        return model

    def predict(self,model,features):
        return model.predict(xgb.DMatrix(features))


