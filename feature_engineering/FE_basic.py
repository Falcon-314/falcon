#========================
# module setting
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns

from sklearn.model_selection import KFold

# =======================
# BaseBlock
# =======================
class BaseBlock(object):
    def __init__(self, CFG):
        self.CFG = CFG

    def fit(self,input_df):
        self.meta_df = input_df
        return self
    
    def transform(self,input_df):
        self.return_df = input_df
        return self
    
    def view(self):
        return self.return_df

    def save(self, filename):
        self.return_df.to_csv(self.CFG.FEATURE_PATH + filename + '.csv',index=False)

# =======================
# Basic Feature Engineering
# =======================

# =======================
# Calculate
# =======================
class CalcBlock(BaseBlock):
   
    def __init__(self, CFG, col1, col2, mode):
        self.CFG = CFG
        self.col1 = col1
        self.col2 = col2
        self.mode = mode
        
    def fit(self,df):
        return self

    def transform(self,df):
        df_tmp = pd.DataFrame(df[self.CFG.ID_col])
        if self.mode == 'plus':
            df_tmp[self.col1 + '_plus_' + self.col2] = df[self.col1] + df[self.col2]
            self.return_df = df_tmp[[self.CFG.ID_col, self.col1 + '_plus_' + self.col2]]
        if self.mode == 'minus':
            df_tmp[self.col1 + '_minus_' + self.col2] = df[self.col1] - df[self.col2]
            self.return_df = df_tmp[[self.CFG.ID_col, self.col1 + '_minus_' + self.col2]]
        if self.mode == 'times':
            df_tmp[self.col1 + '_times_' + self.col2] = df[self.col1] * df[self.col2]
            self.return_df = df_tmp[[self.CFG.ID_col, self.col1 + '_times_' + self.col2]]
        if self.mode == 'devided':
            df_tmp[self.col1 + '_devided_' + self.col2] = df[self.col1] / df[self.col2]
            self.return_df = df_tmp[[self.CFG.ID_col, self.col1 + '_devided_' + self.col2]]
        return self
    
# =======================
# Count
# =======================  
class CountBlock(BaseBlock):
    def __init__(self, CFG, cols, count_name):
        self.CFG = CFG
        self.cols = cols
        self.count_name = count_name

    def fit(self, df):
        return self

    def transform(self, df):
        df_tmp = pd.DataFrame(df[self.CFG.ID_col])
        df_tmp[self.count_name] = df[self.cols].sum(axis = 1)
        self.return_df = df_tmp
        return self

class CountZeroBlock(BaseBlock):
    def __init__(self, CFG, cols, count_name):
        self.CFG = CFG
        self.cols = cols
        self.count_name = count_name

    def fit(self, df):
        return self

    def transform(self, df):
        df_tmp = pd.DataFrame(df[self.CFG.ID_col])
        df_tmp[self.count_name] = (df[self.cols] == 0).sum(axis=1)
        self.return_df = df_tmp
        return self


# =======================
# Aggregation
# =======================
class AggBlock(BaseBlock):
    def __init__(self,CFG, cat, num, agg, name = None):
        self.CFG = CFG
        self.cat = cat
        self.num = num
        self.agg = agg
        if name == None:
            self.name = agg
        else:
            self.name = name
        
    def fit(self, df):
        self.meta_df = df[[self.cat,self.num]].rename(columns = {self.num:self.num + '_' + self.name + '_by_' +  self.cat}).groupby(self.cat).agg(self.agg)
        return self

    def transform(self, df):
        self.return_df = pd.merge(df, self.meta_df, on = self.cat, how = 'left')[[self.CFG.ID_col, self.num + '_' + self.name + '_by_' +  self.cat]]
        return self

# =======================
# Conmination categorical feature; Encodingを実施すること, 引数のデータフレームに特徴量の列が追加される
# =======================
class CombinationBlock(BaseBlock):
    def __init__(self,CFG,col1,col2):
        self.CFG = CFG
        self.col1 = col1
        self.col2 = col2

    def fit(self,df):
        return self
    
    def transform(self,df):
        df[self.col1 + 'and' + self.col2] = df[self.col1].astype('str') + df[self.col2].astype('str')
        return self

# ========================
# binning
# ========================
class BinningBlock(BaseBlock):
    
    def __init__(self,col,edges):
        self.lag = lag
        self.col = col
        self.edges = edges
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[col + '_bin'] = pd.cut(input_df[self.col], edges, labels = False)
        return remain_df[col + '_bin']
   
#---count feature---#
#Flag
class FlagBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df_max = input_df[self.cols].groupby(input_df[self.key]).max()
        _df_min = input_df[self.cols].groupby(input_df[self.key]).min()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Range_')
        return out_df  
    
#flag_cunt
class Flag_count_Block(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df_max = input_df[self.cols].groupby(input_df[self.key]).max()
        _df_min = input_df[self.cols].groupby(input_df[self.key]).min()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Range_')
        return out_df  

#n_distinct : number of unique

#Entropy : apply entropy on frequency table

#freq1name : the number of most frequently appeared category

#freq1ratio : the number of most frequently appeared category / group size


#---time series feature---#
class LagBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].shift(self.lag)
        return output_df.add_prefix('Lag_{}_'.format(self.lag))
    
    
#Lag feature (diff)
class DiffLagBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].diff(self.lag)
        return output_df.add_prefix('DiffLag_{}_'.format(self.lag))

#MeanLag
class MeanLagBlock(BaseBlock):
    
    def __init__(self,window:int,ids,cols):
        self.window = window
        self.ids = ids
        self.cols = cols
        self.meta_df = None
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids,sort = False)[self.cols].rolling(window=self.window).mean().reset_index(drop=True)
        return output_df.add_prefix('MeanLag_{}_'.format(3))

#StdLag
class StdLagBlock(BaseBlock):
    
    def __init__(self,window:int,ids,cols):
        self.window = window
        self.ids = ids
        self.cols = cols
        self.meta_df = None
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].rolling(window=self.window).std().reset_index(drop=True)
        return output_df.add_prefix('StdLag_{}_'.format(self.window))

#SumLag
class SumLagBlock(BaseBlock):
    
    def __init__(self,window:int,ids,cols):
        self.window = window
        self.ids = ids
        self.cols = cols
        self.meta_df = None
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].rolling(window=self.window).sum().reset_index(drop=True)
        return output_df.add_prefix('SumLag_{}_'.format(self.window))

# ==============================
# Encoding
# ==============================

# ==============================
# Frequency Encoding
# ==============================
class FreqBlock(BaseBlock):
    def __init__(self, CFG, freq_col):
        self.CFG = CFG
        self.freq_col = freq_col

    def fit(self, df):
        self.meta_df = df[self.freq_col].value_counts()
        return self

    def transform(self, df):
        df[self.freq_col + '_freq'] = df[self.freq_col].map(self.meta_df)
        self.return_df = df[[self.CFG.ID_col, self.freq_col + '_freq']]
        return self

# ==============================
# Target Encoding
# ==============================
class TargetBlock(BaseBlock):
    def __init__(self, CFG, feature, target):
        self.CFG= CFG
        self.feature= feature
        self.target = target

    def fit_transform(self, train, test):
        self.return_df = {}
        for fold in range(self.CFG.n_fold):
            if fold in self.CFG.trn_fold:
                #t rainとvalidに分割
                trn_idx = train[train['fold'] != fold].index
                val_idx = train[train['fold'] == fold].index
                train_df = train.loc[trn_idx]
                valid_df = train.loc[val_idx]
                        
                # validに対するtarget encoding
                data_tmp = pd.DataFrame({self.feature: train_df[self.feature], 'target': train_df[self.target]})
                target_mean = data_tmp.groupby(self.feature)['target'].mean()
                valid_df.loc[:, self.feature + '_' + self.target + '_target'] = valid_df[self.feature].map(target_mean)

                # trainに対するtarget encoding
                tmp = np.repeat(np.nan, train_df.shape[0])
                kf_encoding = KFold(n_splits=5, shuffle=True, random_state=37)
                for idx_1, idx_2 in kf_encoding.split(train_df, train_df[self.feature]):
                    target_mean = data_tmp.iloc[idx_1].groupby(self.feature)['target'].mean()
                    tmp[idx_2] = train_df[self.feature].iloc[idx_2].map(target_mean)
                train_df.loc[:, self.feature + '_' + self.target + '_target'] = tmp
                
                # train_dfとvalid_dfを結合
                fold_df = pd.concat([train_df[[self.CFG.ID_col, self.feature + '_' + self.target + '_target']], valid_df[[self.CFG.ID_col, self.feature + '_' + self.target + '_target']]], axis = 0)

                # dfの保存
                self.return_df['fold_' + str(fold)] = fold_df

        # testに対するtarget encoding
        data_tmp = pd.DataFrame({self.feature: train[self.feature], 'target': train[self.target]})
        target_mean = data_tmp.groupby(self.feature)['target'].mean()
        test.loc[:, self.feature + '_' + self.target + '_target'] = test[self.feature].map(target_mean)
        test_df = test[[self.CFG.ID_col, self.feature + '_' + self.target + '_target']]

        self.return_df['test'] = test_df

        return self

    def save(self, filename):
        for fold in range(self.CFG.n_fold):
            if fold in self.CFG.trn_fold:
                self.return_df['fold_' + str(fold)].to_csv(self.CFG.FEATURE_PATH + filename + '_fold_' + str(fold) + '.csv',index=False)

        self.return_df['test'].to_csv(self.CFG.FEATURE_PATH + filename + '_test' + '.csv',index=False)
