#debug modeの機能

import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
from catboost import Pool

import wandb

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
   
from wandb.lightgbm import wandb_callback
lgbm_callbacks=[wandb_callback()]

class Lgbm(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        self.model = None

    def fit(self,x_train,y_train,x_valid,y_valid):
        lgb_train = lgb.Dataset(x_train,y_train)
        lgb_valid = lgb.Dataset(x_valid,y_valid)

        model = lgb.train(self.model_params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            early_stopping_rounds=100,
            num_boost_round=10000,
            verbose_eval=False,
            callbacks=[wandb_callback()])
        
        self.model = model

    def predict(self,x_test):
        return self.model.predict(x_test)
    
    def importance(self, features, fold):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = self.model.feature_importance()
        fold_importance_df["fold"] = fold
        return fold_importance_df
        
    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train,x_valid,y_valid)
        oof_df = self.predict(x_valid)
        return oof_df, self.model       

    def visualize(self, importance_df, size = (8,8)):
       cols = (importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)

       best_features = importance_df.loc[importance_df.Feature.isin(cols)]

       plt.figure(figsize=size)
       sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
       plt.title('Features importance (averaged/folds)')
       plt.tight_layout() 
    
   

class Cat(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        self.model = None

    def fit(self,x_train,y_train,x_valid,y_valid):
        train_pool = Pool(x_train,label=y_train,cat_index)
        valid_pool = Pool(x_valid,label=y_valid,cat_index)
        
        model = CatBoost(self.model_params)
        model.fit(train_pool,
                  early_stopping_rounds=50,
                  plot=False,
                  use_best_model=True,
                  eval_set=[valid_pool],
                  verbose=False)
        
        self.model = model

    def predict(self,x_test):
        return self.model.predict(x_test)
        
    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train,x_valid,y_valid)
        oof_df = self.predict(x_valid)
        return oof_df, self.model
    
class Xgb(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        self.model = None
    
    def fit(self,x_train,y_train,x_valid,y_valid):
        xgb_train = xgb.DMatrix(x_train,label=y_train)
        xgb_valid = xgb.DMatrix(x_valid,label=y_valid)

        evals = [(xgb_train,'train'),(xgb_valid,'eval')]

        model = xgb.train(self.model_params,
                         xgb_train,
                         num_boost_round=3000,
                         early_stopping_rounds=50,
                         evals=evals,
                         verbose_eval=False,
                         callbacks=[wandb.xgboost.wandb_callback()]) #custom metricを使用する場合obj = ***を追加
        
        self.model = model
    
    def predict(self,x_test):
        return self.model.predict(xgb.DMatrix(x_test))
        
    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train,x_valid,y_valid)
        oof_df = self.predict(x_valid)
        return oof_df, self.model



