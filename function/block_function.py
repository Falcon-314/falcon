#---module import---#
import torch
import os
import random
import math
import time

import pandas as pd
import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure as figure

from tqdm import tqdm
from contextlib import contextmanager

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

#feature_enginnering run
def run_createfe(input_df,blocks,is_train=False):
    out_df = input_df.copy()
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,is_train)
        _df = func(out_df)
        assert len(_df) == len(out_df),func._name_
        out_df = pd.concat([out_df,_df],axis=1)
    return out_df

#preprocessed run
def run_preprocess(input_df,blocks,is_train=False):
    out_df = input_df.copy()
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,is_train)
        out_df = func(out_df)
    return out_df 

#preprocessed run concat
def run_preprocess_concat(input_train, input_test, blocks):
    input_train['part'] = 'train'
    input_test['part'] = 'test'
    all_df = pd.concat([input_train,input_test],axis = 0).reset_index(drop = True)   
    for block in tqdm(blocks,total=len(blocks)):
        func = get_function(block,True)
        all_df = func(all_df)
    output_train = all_df[all_df['part'] == 'train'].reset_index(drop = True).drop('part', axis = 1)
    output_test = all_df[all_df['part'] == 'test'].reset_index(drop = True).drop('part', axis = 1)
    return output_train, output_test
