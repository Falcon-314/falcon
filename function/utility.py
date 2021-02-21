#---module import---#
import torch
import os
import random
import math
import time

import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from contextlib import contextmanager

#---logging---#
@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def init_logger(OUTPUT_DIR = './'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    log_file=OUTPUT_DIR+'train.log'
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

#---seed settings---#
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#---data loading---#
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#---preprocess---#
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

#select fit or transform
def get_function(block,is_train):
    s = mapping ={
        True:'fit',
        False:'transform'
    }.get(is_train)
    return getattr(block,s)

#feature_enginnering run #return:only preprocessed columns
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

#preprocessed run #return:all columns
def to_preprocessed(input_df,blocks):   
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,True)
        _df = func(input_df)
    return _df    
    
def to_preprocessed_transform(input_df_train,input_df_test,blocks):
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,True)
        _df_train = func(input_df_train)
        func = get_function(block,False)
        _df_test = func(input_df_test)
    return _df_train, _df_test
