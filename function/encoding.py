#モジュールのインポート
import pandas as pd
import numpy as np
from tqdm import tqdm

#BaseBlockを継承して特徴量を作成する
class BaseBlock(object):
    def fit(self,input_df,y=None):
        return self.transform(input_df)
    
    def transform(self,input_df):
        raise NotImplementedError()

#categorical encoding

#Label-Encoding
from sklearn.preprocessing import LabelEncoder
class LabelBlock(BaseBlock):
    
    def __init__(self,col):
        self.meta_df =None
        self.col = col
        self.le = LabelEncoder()
        
    def fit(self,input_df):
        fit_df = input_df.copy()
        fit_df[self.col].fillna('missing',inplace = True)
        self.le.fit(fit_df[self.col])
        return self.transform(input_df)
    
    def transform(self,input_df):
        transform_df = input_df.copy()
        transform_df[self.col].fillna('missing',inplace = True)
        transform_df_enc = self.le.transform(transform_df[self.col])
        return_df = pd.concat([input_df.drop(self.col,axis = 1),transform_df_enc],axis = 1)
        return return_df

#One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
class OneHotBlock(BaseBlock):
    
    def __init__(self,col):
        self.meta_df =None
        self.col = col
        self.ohe = OneHotEncoder(sparse = False,categories = 'auto')
        
    def fit(self,input_df):
        self.ohe.fit(input_df[self.col])
        self.columns = []
        self.columns += [f'{self.col}_{v}' for v in self.ohe.categories[0]]
        return self.transform(input_df)
    
    def transform(self,input_df):
        return_df = pd.concat([input_df.drop(self.col,axis = 1),pd.DataFrame(self.le.transform(input_df[col]),columns = self.columns)],axis = 1)
        return return_df
    
#Frequency Encoding
class FreqBlock(BaseBlock):
    
    def __init__(self,col):
        self.meta_df =None
        self.col = col
        
    def fit(self,input_df):
        self.meta_df = input_df[self.col].value_counts()
        return self.transform(input_df)
    
    def transform(self,input_df): 
        return input_df[self.col].map(self.meta_df)

#Target encoding
from sklearn.model_selection import KFold
def target_encoding(df, target, target_cols, folds):
  
    #trainとtestに分割  
    train = df.query('part == "train"')
    test = df.query('part == "test"')
    
    #最終提出用のデータフレームの保存
    train_save = train.copy()
                    
    # 変数をループしてtarget encoding
    for c in target_cols:
                    
        #事前に作成したfold毎に平均値を作成する
        for fold in folds:
                    
            #trainとvalidに分割
            trn_idx = train[train['fold'] != fold].index
            val_idx = train[train['fold'] == fold].index
            train_df = train.loc[trn_idx]
            valid_df = train.loc[val_idx]
                    
            # validに対するtarget encoding
            data_tmp = pd.DataFrame({c: train_df[c], 'target': train_df[target]})
            target_mean = data_tmp.groupby(c)['target'].mean()
            valid_df.loc[:, c] = valid_df[c].map(target_mean)

            # trainに対するtarget encoding
            # trainもKfoldで分割してtarget encodingする
            tmp = np.repeat(np.nan, train_df.shape[0])
            kf_encoding = KFold(n_splits=5, shuffle=True, random_state=37)
            for idx_1, idx_2 in kf_encoding.split(train_df,train_df[c]):
                target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
                tmp[idx_2] = train_df[c].iloc[idx_2].map(target_mean)
            train_df.loc[:, c] = tmp
            
            #最終的なデータフレームに代入  変数名.isnull().sum()
            train_save.loc[trn_idx, c + f'_fold_{fold}'] = train_df.loc[:, c]
            train_save.loc[val_idx, c + f'_fold_{fold}'] = valid_df.loc[:, c]
            
        # 置換済みのカラムは不要なので削除
        train_save = train_save.drop(c,axis = 1)

    #testのtarget encoding
    for c in target_cols:
                    
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train[c], 'target': train[target]})
        target_mean = data_tmp.groupby(c)['target'].mean()
                    
        # 置換
        for fold in folds:
            test.loc[:, c + f'_fold_{fold}'] = test[c].map(target_mean)
                    
        # 置換済みのカラムは不要なので削除
        test = test.drop(c,axis = 1)
    
    #trainとtestを結合して復元する               
    df = pd.concat([train_save,test],axis = 0).reset_index(drop=True)
                    
    return df

def target_encoding_preprocess(df, target_cols, trn_fold):
    target_cols_encoded = [c + f'_fold_{fold}' for c in target_cols for fold in CFG.trn_fold]
    base_columns = all_df.columns
    for c in target_cols:
        df = df.rename(columns = {c + f'_fold_{fold}':c})
    features = [c for c in base_columns if c not in ['ID', CFG.target_col, 'part', 'fold'] + target_cols_encoded] + target_cols
               
    return folds, features
    
#numerical encoding

#auto-scaling
from sklearn.preprocessing import StandardScaler
def auto_scaling(df, auto_cols):
    #一時的にtrainとtestを分離
    train_tmp = df.query('part == "train"').reset_index(drop = True)
    test_tmp = df.query('part == "test"').reset_index(drop = True)

    #エンコーディングの実行
    scaler =  StandardScaler()
    scaler.fit(train_tmp[auto_cols])
    train_tmp[auto_cols] =  pd.DataFrame(scaler.transform(train_tmp[auto_cols]))
    test_tmp[auto_cols] =  pd.DataFrame(scaler.transform(test_tmp[auto_cols]))

    #trainとtestを再結合
    df = pd.concat([train_tmp,test_tmp],axis = 0).reset_index(drop = True)
    return df

#min-max scaling
from sklearn.preprocessing import MinMaxScaler
def minmac_scaling(df, minmax_cols):
    scaler = MinMaxScaler()
    df[minmax_cols] = scaler.fit_transform(df[minmax_cols])
    return df

#rank gauss:NNだとautoscalingより性能良いらしい
from sklearn.preprocessing import QuantileTransformer
def rankgauss_scaling(df,Rankgauss_cols):
    #一時的にtrainとtestを分離
    train_tmp = df.query('part == "train"').reset_index(drop = True)
    test_tmp = df.query('part == "test"').reset_index(drop = True)

    #エンコーディングの実行
    Rankgauss_scaler = QuantileTransformer(n_quantiles = 100,random_state = 37,output_distribution = 'normal')
    scaler.fit(train_tmp[Rankgauss_cols])
    train_tmp[Rankgauss_cols] =  pd.DataFrame(scaler.transform(train_tmp[Rankgauss_cols]))
    test_tmp[Rankgauss_cols] =  pd.DataFrame(scaler.transform(test_tmp[Rankgauss_cols]))

    #trainとtestを再結合
    df = pd.concat([train_tmp,test_tmp],axis = 0).reset_index(drop = True)
    return df

#対数変換
def log_transform(df, log_cols):
    for col in log_cols:
        df[col] = np.log1p(df[col])
    return df    

#定数倍
def multiple_transform(df, multiple_cols, multiple):
    for col in multiple_cols:
        df[cols] = df[col].apply(lambda x: np.floor(x * multiple))
    return df

#Clipping
def clipping(df,clip_cols,clip_min,clip_max):
    for col in clip_cols:
        df[cols] = df[col].clip(clip_min,clip_max)
    return df
