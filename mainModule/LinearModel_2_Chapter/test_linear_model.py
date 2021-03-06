from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

diabetes_x ,diabetes_y = datasets.load_diabetes(return_X_y=True)
# print(len(diabetes_x),diabetes_x[0:2],diabetes_y[0:2])
# 442 [[ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
#   -0.04340085 -0.00259226  0.01990842 -0.01764613]
#  [-0.00188202 -0.04464164 -0.05147406 -0.02632783 -0.00844872 -0.01916334
#    0.07441156 -0.03949338 -0.06832974 -0.09220405]] [151.  75.]


def regrePlot(x,y,predict_y):
    plt.plot(x,predict_y,color = 'black',label='Prediction')
    plt.scatter(x,y,color = 'blue',label='Raw Data')
    plt.xlabel('Attribute')
    plt.ylabel('Diabete')
    
    plt.show()

regrePlot(diabetes_x,diabetes_y,diabetes_predict_y)


# from .OneAttributeLinearRegressionModel import oneAttributeLinearRegression

# ## 获取单列数据
# diabetes_x = diabetes_x[:,2]

# linear_regression = oneAttributeLinearRegression()
# linear_regression.fit(diabetes_x,diabetes_y)

# diabetes_predict_y = linear_regression.preidct(diabetes_x)


