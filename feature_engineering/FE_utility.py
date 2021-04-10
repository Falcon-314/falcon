import pandas as pd
import numpy as np

#===========================
#BaseBlockの設定
#===========================
class BaseBlock(object):
    def __init__(self, CFG):
        self.CFG = CFG

    def fit(self,input_df):
        self.meta_df = input_df
        return self
    
    def transform(self,input_df):
        return self.meta_df

    def save(self, filename):
        self.meta_df.to_feather(CFG.FEATURE_PATH + filename + '.ftr')
