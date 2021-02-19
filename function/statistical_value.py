# 最小値と最大値の差
def range_diff(x):
    return x.max() - x.min()

#平均に対する割合を計算
def ratio(x):
    return x / x.mean()

#平均に対する差を計算
def mean_diff(x):
    return x - x.mean()

# 第一/三四分位点とその差
def third_quartile(x):
    return x.quantile(0.75)

def first_quartile(x):
    return x.quantile(0.25)

def quartile_range(x):
    return x.quantile(0.75) - x.quantile(0.25)
