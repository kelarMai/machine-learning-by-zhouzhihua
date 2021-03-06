## ndarray 基本函数测试
import numpy as np
a = [[1,2,3,4],[4,5,6,7],[7,8,9,10]]
a_array = np.array(a)
# b = [1,2,3]
# b_array = np.array(b)
# print(a_array.T.dot(b))
# print(a_array.ndim)
# print(np.power(b_array,2))
# print(b_array**2)
# print(a_array.sum(axis=0))
# print(2*b_array)

# x = np.array(range(1,101,1))
# x = x.reshape(10,10)
# y = x.dot(np.array([1,3,11,0.2,3,1,5,2,10,0.02]))

# print(type(b_array.mean(axis=0)))
# print(b_array.var(axis=0,ddof=1))

# print(b_array.ndim)

# print(isinstance(a_array,np.ndarray))



## a_test[:,i] = a_temp 广播方法的测试，这里重点是 dtype 需要设定
# a_test = np.empty_like(a_array,dtype=float)
# for i in range(a_array.shape[1]):
#     a_temp = np.true_divide(a_array[:,i] - i,2000)
#     print(a_temp)
#     a_test[:,i] = a_temp 
# print(a_test)




## 测试 scipy 的最小二乘法使用
# import numpy as np
# from scipy.optimize import leastsq

# x = np.array(range(1,101,1))
# x = x.reshape(10,10)
# y = x.dot(np.array([1,3,11,0.2,3,1,5,2,10,0.02]))

# def Func(p,x):
#     a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 = p
#     return a1*x[0] + a2*x[1] + a3*x[2] + a4*x[3] + a5*x[4] + a6*x[5] + a7*x[6] + a8*x[7] + a9*x[8] + a10*x[9]

# def error(p,x,y):
#     return (Func(p,x) - y)**2

