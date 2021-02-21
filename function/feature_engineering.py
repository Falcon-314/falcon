#---module setting---#
import pandas as pd
import numpy as np
    
#----basic feature engineering---#

#calculate
class CalcBlock(BaseBlock):
    
    def __init__(self,col1,col2,mode):
        self.col1 = col1
        self.col2 = col2
        self.mode = mode
        
    def fit(self,input_df):
        return self.transform(input_df)

    def transform(self,input_df):
        remain_df = input_df.copy()
        if mode == 'plus':
            remain_df[self.col1 + '_plus_' + self.col2] = input_df[self.col1] + input_df[self.col2]
            return_df = remain_df[self.col1 + '_plus_' + self.col2]
        if mode == 'minus':
            remain_df[self.col1 + '_minus_' + self.col2] = input_df[self.col1] - input_df[self.col2]
            return_df = remain_df[self.col1 + '_minus_' + self.col2]
        if mode == 'multiple':
            remain_df[self.col1 + '_multiple_' + self.col2] = input_df[self.col1] * input_df[self.col2]
            return_df = remain_df[self.col1 + '_multiple_' + self.col2]
        if mode == 'devide':
            remain_df[self.col1 + '_devide_' + self.col2] = input_df[self.col1] / input_df[self.col2]
            return_df = remain_df[self.col1 + '_devide_' + self.col2]
        return return_df
    
#Aggregation


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

