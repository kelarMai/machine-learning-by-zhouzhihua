import numpy as np
from ..util.lsqr import lsqr

class MultiAttributeLinearRegression():
    def __init__(self):
        self.coefficients = None
        pass

    def fit(self,x,y):
        '''
        :param x 二维 ndarray 类型，为输入的数据量，包含截距
            例如    [[x11,x12,x13,1]
                    [x21,x22,x23,1]
                    [x31,x32,x33,1]]
        :param y 一维 ndarray 类型，为对应的输出数据；
        —————————————————————————————————————————————————
        系数不分开设置为变量系数或者常系数；
        '''
        is_sparse_matrix = False

        x_rank = np.linalg.matrix_rank(x)
        # print(f"X.shape:{x.shape};X.rank:{x_rank}")
        x_features = len(x[0])
        if x_rank < x_features:
            is_sparse_matrix = True

        if is_sparse_matrix == False:
            ## 矩阵的秩等于属性数，np.dot(x.T,x)为满秩矩阵
            ## 满秩矩阵可以通过直接求解一次导数来求解系数
            interin_invert_matrix = np.linalg.inv(x.T.dot(x))
            self.coefficients = interin_invert_matrix.dot(x.T).dot(y)
        else :
            self.coefficients,error = lsqr(x,y,alpha=0.001,error = 0.01,count = 1000)
            print(self.coefficients)
        pass

    def predict(self,x):
        '''
        :param x 数据向量， ndarray 类型
        
        :return 预测的 y 值
        '''
        if isinstance(self.coefficients,np.ndarray) != True:
            raise EOFError("还没有进行模型拟合")
        return x.dot(self.coefficients)