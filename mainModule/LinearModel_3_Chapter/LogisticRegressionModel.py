import numpy as np


class LogisticRegression():
    '''
    参考 《西瓜书》.P59,60 页
    '''
    def __init__(self):
        self.beta = None
        pass

    def LogLikelihood(self,X_hat,Y,beta):
        '''
        极大似然值
        '''
        Z = X_hat.dot(beta)
        probability = -(Z * Y) + np.log( 1 + np.exp(Z))
        probability = probability.sum()
        return probability
        pass

    def PosterioriProbability_1(self,X,beta):
        
        def sigmoid(z):
            '''
            :param z 需要转化为 (0-1) 区间的数据
            '''
            return 1 / (1 + np.exp(-z))

        Z = X.dot(beta)
        return sigmoid(Z)

    def gradDescFit(self,X,Y,beta,alpha,count):
        X_hat = np.c_[X,np.ones(X.shape[0])]
        grad = np.empty_like(X_hat)

        for i in range(count):
            temp_diff = Y - self.PosterioriProbability_1(X_hat,beta)
            for i in range(X.shape[0]):
                grad[i] = X_hat[i] * temp_diff[i]
            grad_sum = grad.sum(axis=0)

            beta = beta - alpha * grad_sum

            if i % 10 == 0:
                probability = self.LogLikelihood(X_hat,Y,beta)
                print(f'第 {i} 次迭代的似然概率为 {probability}')
        pass

    def newtonDesc(self,X,Y,beta,alpha,count):

        pass

    def fit(self,X,Y,method = 'gradDesc',alpha = 0.01,count = 100):
        '''
        :param X 用来拟合的二维 ndarray 矩阵，包含一列为1的列用来拟合截距
        :param Y 用来拟合的一维 ndarray 向量
        :param method 可选参数 "gradDesc" 和 "newtonDesc"
        :param alpha 下降率
        :param count 最多迭代次数        
        '''
        beta = np.zeros(X.shape[1] + 1)
        func = None

        if method == 'gradDesc':
            func = self.gradDescFit
        elif method == 'newtonDesc':
            func = self.newtonDesc

        func(X,Y,beta,alpha,count)
        self.beta = beta

        pass

    def predict(self,X):
        '''
        
        '''
        pass