#module setting
import pandas as pd
import numpy as np
from tqdm import tqdm

#BaseBlock
class BaseBlock(object):
    def fit(self,input_df,y=None):
        return self.transform(input_df)
    
    def transform(self,input_df):
        raise NotImplementedError()
        
#WrapperBlock
class WrapperBlock(BaseBlock):
    def __init__(self,function):
        self.function=function
    
    def transform(self,input_df):
        return self.function(input_df)

#fit or transform
def get_function(block,is_train):
    s = mapping ={
        True:'fit',
        False:'transform'
    }.get(is_train)
    return getattr(block,s)

#feature_enginnering run
def to_feature(input_df,remain_df,blocks):
    out_df = remain_df
    
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,True)
        _df = func(input_df)
        assert len(_df) == len(input_df),func._name_
        out_df = pd.concat([out_df,_df],axis=1)
    return out_df    
    
def to_feature_transform(input_df_train,remain_df_train,input_df_test,remain_df_test,blocks):
    out_df_train = remain_df_train
    out_df_test = remain_df_test
    
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,True)
        _df_train = func(input_df_train)
        assert len(_df_train) == len(input_df_train),func._name_
        func = get_function(block,False)
        _df_test = func(input_df_test)
        assert len(_df_test) == len(input_df_test),func._name_
        out_df_train = pd.concat([out_df_train,_df_train],axis=1)
        out_df_test = pd.concat([out_df_test,_df_test],axis=1)
    return out_df_train, out_df_test  
    
 #----basic feature engineering---#

#plus
class PlusBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.col1 = col1
        self.col2 = col2
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + '_plus_' + self.col2] = input_df[self.col1] + input_df[self.col2]
        return remain_df[self.col1 + '_plus_' + self.col2]

#minus
class MinusBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.col1 = col1
        self.col2 = col2
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + '_minus_' + self.col2] = input_df[self.col1] - input_df[self.col2]
        return remain_df[self.col1 + '_minus_' + self.col2]
    
#multiple
class MultipleBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.col1 = col1
        self.col2 = col2
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + '_mulpiple_' + self.col2] = input_df[self.col1] * input_df[self.col2]
        return remain_df[self.col1 + '_multiple_' + self.col2]

#devide
class DevideBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.col1 = col1
        self.col2 = col2
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + '_devide_' + self.col2] = input_df[self.col1] / input_df[self.col2]
        return remain_df[self.col1 + '_devide_' + self.col2]    
    
#Aggregation_mean
class MeanBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).mean()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Mean_')
        return out_df    
    
#Aggregation_std
class StdBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).std()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Std_')
        return out_df
    
#Aggregation_skew
class SkewBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).skew()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Std_')
        return out_df
    
#ratio
class RatioBlock(BaseBlock):
    
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
    
#Aggregation_sum
class SumBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).sum()
        _df = (_df.T / _df.sum(axis=1)).T #standardize
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Sum_')
        return out_df    

#Aggregation_max
class MaxBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).max()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Max_')
        return out_df

#Aggregation_min
class MinBlock(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).min()
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Max_')
        return out_df    
    
#Aggregation_range
class RangeBlock(BaseBlock):
    
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
    
#Aggregation_quantile
class Agg_QuantileBlock(BaseBlock):
    
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
        out_df = out_df.add_prefix('Agg_Range_')
        return out_df  
    
#quantilerange
class Agg_QuantileRange_Block(BaseBlock):
    
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
    
#Aggregation_count
class Agg_Count_Block(BaseBlock):
    
    def __init__(self,key:str,cols):
        self.key =key
        self.meta_df =None
        self.cols = cols
        
    def fit(self,input_df):
        _df = input_df[self.cols].groupby(input_df[self.key]).count()
        _df = (_df.T / _df.sum(axis=1)).T #standardize
        self.meta_df = _df
        return self.transform(input_df)
    
    def transform(self,input_df):
        out_df = pd.merge(input_df[self.key],self.meta_df,on=self.key,how='left').drop(columns=[self.key])
        out_df = out_df.add_prefix('Agg_Count_')
        return out_df    
    
#conmination categorical feature
class CombinationBlock(BaseBlock):
    
    def __init__(self,col1,col2):
        self.lag = lag
        self.col1 = col1
        self.col2 = col2

    def fit(self,input_df):
        return self.transform(input_df)
    
    def transform(self,input_df):
        remain_df = input_df.copy()
        remain_df[self.col1 + 'and' + self.col2] = input_df[self.col1].astype('str') + input_df[self.col2].astype('str')
        return remain_df[self.col1 + 'and' + self.col2]
       
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
    
#Lag feature
class LagBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].lag(self.lag)
        return output_df.add_prefix('Lag_{}_'.format(self.lag))
    
    
#Lag feature (diff)
class Lag_DiffBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        output_df = input_df.groupby(self.ids)[self.cols].diff(self.lag)
        return output_df.add_prefix('Lag_{}_'.format(self.lag))

#flag
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
