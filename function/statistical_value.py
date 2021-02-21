# 最小値と最大値の差
def range_diff(x):
    return x.max() - x.min()

# 最小値と最大値の比
def range_ratio(x):
    return x.max() / x.min()

#平均に対する割合を計算
def ratio(x):
    return x / x.mean()

#平均に対する差を計算
def mean_diff(x):
    return x - x.mean()

#平均と偏差の比
def mean_variance(x):
    return x.std() / x.mean()

#Z score
def z_score(x):
    return ( x - x.mean() ) / x.std()

# 第一四分位点
def third_quartile(x):
    return x.quantile(0.75)

#第三四分位点
def first_quartile(x):
    return x.quantile(0.25)

#第一四分位点と第三四分位点の差
def quartile_range(x):
    return x.quantile(0.75) - x.quantile(0.25)

#平均値より大きい値の数と小さい値の数の比
def hl_ratio(x):
    um(ifelse(x > mean(x),1,0)) / sum(ifelse(x >= mean(x),0,1))
    
#MAD : Median Absolute Deviation : median( |x - median(x)| )

#Beyond1std : Calculating the ratio beyond 1 std

#Shapiro-Wilk Statistic

