import pandas as pd
import lightgbm as lgb
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from abc import abstractmethod
class Base_Model(object):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError

class LgbmClass(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def train(self, CFG, x_train, y_train, x_valid, y_valid):
            
            model = lgb.LGBMClassifier(**CFG.lgbm_params)
            model.fit(x_train, y_train,
                  eval_set=(x_valid, y_valid),
                  eval_metric='logloss',
                  verbose=100,
                  early_stopping_rounds=500
                  )
            
            return model

    def valid(self, CFG, x_valid, model):
            preds = model.predict_proba(x_valid)
            return preds

    def inference(self, CFG, x_test, model):
            preds = model.predict_proba(x_test)
            return preds

class Lgbm(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def train(self, CFG, x_train, y_train, x_valid, y_valid):
        
            lgb_train = lgb.Dataset(x_train,y_train)
            lgb_valid = lgb.Dataset(x_valid,y_valid)
            
            model = lgb.train(self.model_params,
                            train_set=lgb_train,
                            valid_sets=[lgb_valid],
                            valid_names=['valid'],
                            early_stopping_rounds=500,
                            num_boost_round=50000,
                            verbose_eval=100)
            
            return model

    def valid(self, CFG, x_valid, model):
            preds = model.predict(x_valid)
            return preds

    def inference(self, CFG, x_test, model):
            preds = model.predict(x_test)
            return preds        
      
def visualize_importance(CFG, features, MODEL_NAME, size):
    importance_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            model = pickle.load(open(CFG.MAIN_PATH + f'{MODEL_NAME}' + f'_fold_{fold}.sav','rb'))
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = features
            fold_importance_df["importance"] = model.feature_importance()
            fold_importance_df["fold"] = fold
            importance_df = pd.concat([importance_df, fold_importance_df])

    cols = (importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)

    best_features = importance_df.loc[importance_df.Feature.isin(cols)]

    plt.figure(figsize=size)
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
