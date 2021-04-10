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
        self.return_df = input_df
        return self
    
    def view(self):
        return self.return_df

    def save(self, filename):
        self.return_df.to_csv(self.CFG.FEATURE_PATH + filename + '.csv')
