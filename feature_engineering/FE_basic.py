#========================
# module setting
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns

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
# Feature Loading
# =======================
def feature_load(CFG, df, file_name):
    load_df = pd.read_csv(CFG.FEATURE_PATH + file_name)
    df = pd.merge(df,load_df,on = CFG.ID_col, how = 'left')
    return df

# =======================
# Basic Feature Engineering
# =======================

# =======================
# Calculate
# =======================
class CalcBlock(BaseBlock):
   
    def __init__(self,col1,col2,mode):
        self.col1 = col1
        self.col2 = col2
        self.mode = mode
        
    def fit(self,df):
        if self.mode == 'plus':
            df[self.col1 + '_plus_' + self.col2] = df[self.col1] + df[self.col2]
            self.meta_df = df[[CFG.ID_col, self.col1 + '_plus_' + self.col2]]
        if self.mode == 'minus':
            df[self.col1 + '_minus_' + self.col2] = df[self.col1] - df[self.col2]
            self.meta_df = df[[CFG.ID_col, self.col1 + '_minus_' + self.col2]]
        if self.mode == 'times':
            df[self.col1 + '_times_' + self.col2] = df[self.col1] - df[self.col2]
            self.meta_df = df[[CFG.ID_col, self.col1 + '_times_' + self.col2]]
        if self.mode == 'devided':
            df[self.col1 + '_devided_' + self.col2] = df[self.col1] - df[self.col2]
            self.meta_df = df[[CFG.ID_col, self.col1 + '_devided_' + self.col2]]
        return self

    def transform(self,df):
        if self.mode == 'plus':
            self.return_df = pd.merge(df, self.meta_df, on = CFG.ID_col, how = 'left')[[CFG.ID_col,self.col1 + '_plus_' + self.col2]]
        if self.mode == 'minus':
            self.return_df = pd.merge(df, self.meta_df, on = CFG.ID_col, how = 'left')[[CFG.ID_col,self.col1 + '_minus_' + self.col2]]
        if self.mode == 'times':
            self.return_df = pd.merge(df, self.meta_df, on = CFG.ID_col, how = 'left')[[CFG.ID_col,self.col1 + '_times_' + self.col2]]
        if self.mode == 'devided':
            self.return_df = pd.merge(df, self.meta_df, on = CFG.ID_col, how = 'left')[[CFG.ID_col,self.col1 + '_devided_' + self.col2]]
        return self
      
# =======================
# Aggregation
# =======================
class AggBlock(BaseBlock):
    def __init__(self,CFG,key:list,col,agg_setting,agg_name):
        self.CFG = CFG
        self.key = key
        self.col = col
        self.agg_setting = agg_setting
        self.agg_name = agg_name

    def fit(self,df):
        #集計の実行
        use_cols = [self.CFG.ID_col, self.col] + self.key
        columns_name = [self.CFG.ID_col] + ["_".join([self.col, self.agg_name] + self.key)] +self.key
        self.meta_df = df[use_cols].groupby(self.key).agg(self.agg_setting).reset_index()
        self.meta_df.columns = columns_name
        return self
    
    def transform(self,df):
        self.return_df = pd.merge(df, self.meta_df, on = CFG.ID_col, how = 'left')
        return self

# =======================
# Conmination categorical feature
# =======================
class CombinationBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.col1 = col1
        self.col2 = col2

    def fit(self,input_df):
        self.meta_df = input_df
        return self
    
    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + 'and' + self.col2] = input_df[self.col1].astype('str') + input_df[self.col2].astype('str')
        return self
       
#binning
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

#---merge additional data---#
class Agg_MergeBlock(BaseBlock):
    def __init__(self,key:list,cols:list,agg_settings:list, add):
        self.key = key
        self.meta_df =None
        self.columns_name = None
        self.cols = cols
        self.agg_settings =agg_settings
        self.add = add

    def fit(self,df):
        #集計の実行
        _add = self.add.groupby(self.key)[self.cols].agg(self.agg_settings).reset_index()

        #列名の変更
        _aggs = []
        for agg in self.agg_settings:
            if isinstance(agg, str):
                _aggs.append(agg)
            else:
                _aggs.append(agg.__name__)
        self.columns_name = self.key + ["_".join(self.key+[c,agg]) for c in self.cols for agg in _aggs]
        _add.columns = self.columns_name
        self.meta_df = _add
        return self.transform(df)
    
    def transform(self,df):
        df = df.merge(self.meta_df, on=self.key, how='left')
        return df[self.columns_name].drop(self.key, axis = 1)

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

