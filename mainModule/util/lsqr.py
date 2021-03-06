import numpy as np

class DataNormalization():
    '''
    中心化有多种方法
        1. x = (x - min)/(max-min)
        2. x = x/max
        3. x = (x - mean)/variance
    参考：https://www.cnblogs.com/xuezou/p/9332763.html
    这里使用第三种
    '''
    def __init__(self):
        # 均值
        self.mean = None
        # 方差
        self.variance = None
    
    def fit(self,X):
        '''
        :param X 初始化归一化参数的 ndarray

        如果 X 是二维矩阵，mean 和 variance 是一维向量
        如果 X 是一维向量，mean 和 variance 是常数
        '''
        self.mean = X.mean(axis=0)
        self.variance = X.var(axis=0,ddof=1)

        return self
    
    def transfrom(self,X):
        '''
        :param X 根据已有的归一化数据，进行归一化的 ndarray
        :return X_rev 已经归一化完成的 ndarray
        '''
        X_rev = np.empty_like(X,dtype=float)
        if X.ndim == 1:
            X_rev = (X - self.mean) / self.variance
        else :
            for col in range(X.shape[1]):
                X_rev[:,col] = np.true_divide((X[:,col] - self.mean[col]) , self.variance[col])
        
        return X_rev

    def reTransform(self,X):
        '''
        把标准化后的值恢复
        :param X 需要修复的标准化值
        '''
        if isinstance(self.mean,np.ndarray):
            # X 为变量向量 ndarray 类型
            return np.multiply(X,self.variance) + self.mean
        else:
            # X 为单个参数
            return X * self.variance + self.mean



def lsqr(A,b,alpha = 0.01,error = 0.01,count = 100000):
    '''
    使用梯度下降法求解 min ||Ax-b||^2 类似问题
    参考1： https://zhuanlan.zhihu.com/p/36493862
    :param A 任意秩的矩阵
    :param b Ax = b 的右边数据，一般为向量格式
    :param alpha 梯度下降的下降速度
    :error 最小二乘法的最小误差
    :count 梯度下降的次数

    return X 为参数向量,error_temp 为误差
    '''
    ## 初始化参数
    X = np.ones_like(A[0])
    count_cumulate = 1
    error_temp = 0
    A_T = A.T

    # print(f"A 的值 {A}；\nb 的值 {b}")
    ## 求导数，然后梯度下降
    temp_derivate = np.zeros_like(X)
    for t in range(count):
        ## 导数
        temp_AX_b = A.dot(X) - b
        temp_derivate = alpha*(A_T.dot(temp_AX_b))
        # print(f"temp_derivate: {temp_derivate}")
        X = X - temp_derivate
        
        ## 计算误差
        error_temp = (A.dot(X) - b)**2
        # print(f"误差序列为 {error_temp}")
        error_temp = error_temp.sum()
        # print(f"第{count_cumulate}次迭代，X为：{X}，误差为：{error_temp}；")
        # print("\n")
        if error_temp <= error:
            return X,error_temp

    return  X,error_temp
