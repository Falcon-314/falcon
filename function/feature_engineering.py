#---module setting---#
import pandas as pd
import numpy as np

#BaseBlock
class BaseBlock(object):
    def fit(self,input_df,y=None):
        return self.transform(input_df)
    
    def transform(self,input_df):
        raise NotImplementedError()

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
        if self.mode == 'plus':
            remain_df[self.col1 + '_plus_' + self.col2] = input_df[self.col1] + input_df[self.col2]
            return_df = remain_df[self.col1 + '_plus_' + self.col2]
        if self.mode == 'minus':
            remain_df[self.col1 + '_minus_' + self.col2] = input_df[self.col1] - input_df[self.col2]
            return_df = remain_df[self.col1 + '_minus_' + self.col2]
        if self.mode == 'multiple':
            remain_df[self.col1 + '_multiple_' + self.col2] = input_df[self.col1] * input_df[self.col2]
            return_df = remain_df[self.col1 + '_multiple_' + self.col2]
        if self.mode == 'devide':
            remain_df[self.col1 + '_devide_' + self.col2] = input_df[self.col1] / input_df[self.col2]
            return_df = remain_df[self.col1 + '_devide_' + self.col2]
        return return_df
    
#Aggregation
class AggBlock(BaseBlock):
    def __init__(self,key:str,cols,agg_settings):
        self.key = key
        self.meta_df =None
        self.columns_name = None
        self.cols = cols
        self.agg_settings = agg_settings

    def fit(self,input_df):
        #集計の実行
        _add = input_df[[self.key] + self.cols].groupby(self.key).agg(self.agg_settings).reset_index()

        #列名の変更
        _aggs = []
        for agg in self.agg_settings:
            if isinstance(agg, str):
                _aggs.append(agg)
            else:
                _aggs.append(agg.__name__)
        self.columns_name = [self.key] + ["_".join([c, agg, self.key]) for c in self.cols for agg in _aggs]
        _add.columns = self.columns_name

        self.meta_df = _add
        return self.transform(input_df)
    
    def transform(self,input_df):
        output_df = input_df.merge(self.meta_df, on=self.key, how='left')
        return output_df[self.columns_name].drop(self.key, axis = 1)

class AggCalcBlock(BaseBlock):
    def __init__(self,key:str,col,agg,mode):
        self.key = key
        self.columns_name = None
        self.col = col
        self.agg = agg
        self.mode = mode

    def fit(self,input_df):
        return self.transform(input_df)
    
    def transform(self,input_df):
        output_df = pd.DataFrame()
        if self.mode == 'ratio':
            output_df[self.col + '_ratio_' + f'{self.key}'] =  input_df[self.col] / input_df[self.col + f'_{self.agg}_' + f'{self.key}']
            return_df = output_df[self.col + '_ratio_' + f'{self.key}']
        if self.mode == 'diff':
            output_df[self.col + '_diff_' + f'{self.key}'] =  input_df[self.col] - input_df[self.col + f'_{self.agg}_' + f'{self.key}']
            return_df = output_df[self.col + '_diff_' + f'{self.key}']
        return return_df    
    

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

#---merge additional data---#
class Agg_MergeBlock(BaseBlock):
    def __init__(self,key:str,cols,agg_settings, add):
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
        self.columns_name = [self.key] + ["_".join([c, agg, self.key]) for c in self.cols for agg in _aggs]
        _add.columns = self.columns_name

        self.meta_df = _add
        return self.transform(df)
    
    def transform(self,df):
        df = df.merge(self.meta_df, on=self.key, how='left')
        return df[self.columns_name].drop(self.key, axis = 1)

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

#MeanLag
class MeanLagBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols
        self.meta_df = None
        
    def fit(self,input_df):
        tmp = input_df.copy()
        tmp[self.ids] = tmp[self.ids] + self.lag
        tmp = tmp.rename(columns = {self.cols : self.cols + '_meanLag'})
        tmp = tmp.groupby(self.ids)[self.cols + '_meanLag'].mean()
        self.meta_df = tmp
        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df,self.meta_df, on = self.ids, how = 'left')[self.cols + '_meanLag']

#StdLag
class MeanLagBlock(BaseBlock):
    
    def __init__(self,lag:int,ids,cols):
        self.lag = lag
        self.ids = ids
        self.cols = cols
        self.meta_df = None
        
    def fit(self,input_df):
        tmp = input_df.copy()
        tmp[self.ids] = tmp[self.ids] + self.lag
        tmp = tmp.rename(columns = {self.cols : self.cols + '_stdLag'})
        tmp = tmp.groupby(self.ids)[self.cols + '_stdLag'].std()
        self.meta_df = tmp
        return self.transform(input_df)

    def transform(self,input_df):
        return pd.merge(input_df,self.meta_df, on = self.ids, how = 'left')[self.cols + '_stdLag']
