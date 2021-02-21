# 最小値と最大値の差
def range_diff(x):
    return x.max() - x.min()

# 最小値と最大値の比
def range_ratio(x):
    return x.max() / x.min()

#平均と偏差の比
def mean_variance(x):
    return x.std() / x.mean()

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
    
#MAD : Median Absolute Deviation : median( |x - median(x)| )

#Beyond1std : Calculating the ratio beyond 1 std

#Shapiro-Wilk Statistic

