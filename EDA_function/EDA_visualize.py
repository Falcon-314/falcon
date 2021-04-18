import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns


#==============================
#目的変数と特徴量の関係の可視化
#==============================
def EDA_scatter(CFG, train, feature, target):
    sns.scatterplot(train[feature], train[target])
    corr_score = np.corrcoef(train[feature], train[target])[0,1]
    print('相関係数：', corr_score)
    
    
#==============================
#多変量解析(ヒートマップ)
#==============================
def EDA_heatmap(CFG, df, size):
    plt.figure(figsize=size)
    sns.heatmap(data=df.corr().round(2), annot=True, cmap='coolwarm', linewidths=1, square=True)

def corr_map(data,collist):
    plt.figure(figsize=(12, 12)) 
    sns.heatmap(data = data[collist].corr().round(2),annot=True,cmap='coolwarm',linewidths=0.2,square=True)
    
#=============================
#カテゴリーごとの統計量
#=============================
def statistical_cat(CFG, df, feature, category):
    df_ = pd.DataFrame()
    for cat in train[category].value_counts().index:
        df_tmp = train.loc[train[category]==cat, feature].describe().rename(cat)
        df_ = pd.concat([df_, df_tmp],axis = 1)
    df_tmp = train[feature].describe().rename('ALL')
    df_ = pd.concat([df_, df_tmp],axis = 1)
    return df_

  #======================================
#カテゴリーごとのヒストグラム
#======================================
def distplot_cat(CFG, df, feature, category):
    for cat in df[category].value_counts().index:
        sns.distplot(df.loc[df[category] == cat, feature], label=cat)
    plt.legend(loc="best")
    plt.show()

#==============================
#目的変数と特徴量の関係の可視化(カテゴリー別)
#==============================
def EDA_scatter_cat(CFG, train, feature, target, category, category_name):
    sns.scatterplot(train.loc[train[category] == category_name, feature], train.loc[train[category] == category_name, target])
    corr_score = np.corrcoef(train.loc[train[category] == category_name, feature], train.loc[train[category] == category_name, target])[0,1]
    print('相関係数：', corr_score)

#======================================
#カテゴリーごとの散布図
#======================================
def scatterplot_cat(CFG, df, feature, category, target):
    for cat in df[category].value_counts().index:
        sns.scatterplot(df.loc[df[category] == cat, feature],df.loc[df[category] == cat, target], label=cat)
    plt.legend(loc="best")
    plt.show()
    
#==========================
#外れ値の可視化(カテゴリー別)
#==========================
def stripplot_cat(CFG, df, feature, category):
    sns.stripplot(x=category, y=feature,data = df)

#======================================
#カテゴリーごとのヒストグラム
#======================================
def distplot_cat(CFG, df, feature, category):
    for cat in df[category].value_counts().index:
        sns.distplot(df.loc[df[category] == cat, feature], label=cat)
    plt.legend(loc="best")
    plt.show()

#=============================
#欠損値の有無ごとのヒストグラム
#=============================
def distplot_nan(CFG, df, feature):
    sns.distplot(df.loc[df[feature].isnull(), feature], label = 'missing')
    sns.distplot(df.loc[~df[feature].isnull(), feature], label = 'not missing')
    plt.legend(loc="best")
    plt.show()

#=============================
#欠損値の有無ごとのヒストグラム(カテゴリー別)
#=============================
def distplot_nan_cat(CFG, df, feature, category):
    df_tmp = df.loc[df[feature].isnull()]
    for cat in df_tmp[category].value_counts().index:
        sns.distplot(df_tmp.loc[df_tmp[category] == cat, feature], label=cat)
    plt.legend(loc="best")
    plt.show()

#==========================
#外れ値の可視化
#==========================
def boxplot(CFG, df, feature):
    sns.boxplot(df[feature])

#==========================
#外れ値の可視化(カテゴリー別)
#==========================
def boxplot_cat(CFG, df, feature, category):
    sns.boxplot(x=category, y=feature,data = df)

#=================================
#正規性の確認
#=================================
import scipy.stats as stats
def qqplot(CFG, df, feature):
    stats.probplot(df[feature],dist='norm',plot=plt) 
    plt.show()

#===========================
#trainとtestの比較(ベン図)
#===========================
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os
from tqdm import tqdm_notebook as tqdm

def draw_venn(CFG, df1, df2, save):
    
    df1_col = df1.columns.tolist()
    df2_col = df2.columns.tolist()
    
    if len(df1_col) > len(df2_col):
        df_col = df2_col
    else:
        df_col = df1_col
    
    plt.figure(figsize=(20,20), facecolor='w')
    
    c = 4
    r = (len(df_col) // c) + 1
    
    for i, col in tqdm(enumerate(df_col)):
        plt.subplot(r, c, i+1)
        s1 = set(df1[col].unique().tolist())
        s2 = set(df2[col].unique().tolist())
        venn2(subsets=[s1, s2], set_labels=['Train', 'Test'])
        plt.title(str(col), fontsize=14)

    if save:
        plt.savefig(CFG.MAIN_PATH + 'venn.png', bbox_inches='tight')

    plt.show()
    
    return df_col
    
#===========================
#trainとtestの比較(ヒストグラム)
#===========================
def dist_compare(CFG, df1, df2, feature):
    sns.distplot(df1[feature], label = 'df1')
    sns.distplot(df2[feature], label = 'df2')
    plt.legend(loc="best")
    plt.show()

