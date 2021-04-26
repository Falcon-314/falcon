import pandas as pd
import numpy as np
        
# =======================
# Feature Loading
# =======================
def feature_load(CFG, df, file_name):
    load_df = pd.read_csv(CFG.FEATURE_PATH + file_name)
    df = pd.merge(df,load_df,on = CFG.ID_col, how = 'left')
    return df
