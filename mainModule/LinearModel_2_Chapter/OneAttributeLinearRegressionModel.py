import numpy as np

class oneAttributeLinearRegression():
    def __init__(self):
        self.coefficient = 0
        self.intercept = 0

    def fit(self,x,y):
        '''
        :param x 数据向量，ndarray 类型
        :param y 结果向量，ndarray 类型
        
        最小二乘法求一次导数等于0，获取系数和截距的最优解
        '''
        x_sum = x.sum()
        x_mean = x.mean()
        x_sum_power = np.power(x.sum(),2)
        x_power_sum = (np.power(x,2)).sum()
        array_len = len(x)
        y_sum = y.sum()
        self.coefficient = ( np.dot(y.T,x) - (y * x_mean).sum() ) / ( x_power_sum -  np.true_divide(x_sum_power,array_len))
        self.intercept = (y_sum - (self.coefficient * x).sum() ) / array_len
        pass

    def preidct(self,x):
        '''
        :param x 数据向量，ndarray 类型
        
        :return 预测的 y 值
        '''
        if self.coefficient == 0 and self.intercept == 0:
            raise EOFError("还没有进行模型拟合")
        return x * self.coefficient + self.intercept