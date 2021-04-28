#モジュールのインポート
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================
# Categorical Feature
# ================================

# ================================
# Label-Encoding
# ================================
from sklearn.preprocessing import LabelEncoder
def label_encoding(df,label_cols):
    for col in label_cols:
        df[col].fillna('missing',inplace = True) #欠損値があるとエンコードできないので一度missingに変換
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df
 
# ===============================
# One-Hot Encoding
# ===============================


# ===============================
# Hash Encoding
# ===============================

# ================================
# Numerical Feature
# ================================

# auto-scaling

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
