import sys 
f = open('./mainModule/util/log.txt','w',encoding='utf-8')
sys.stdout = f

# from .lsqr import lsqr,DataNormalization
import numpy as np
from scipy.optimize import leastsq


# x_nor = DataNormalization()
# x_nor.fit(x)
# x_rev = x_nor.transfrom(x)


# y_nor = DataNormalization()
# y_nor.fit(y)
# y_rev = y_nor.transfrom(y)

# lsqr(x,y,alpha=0.01,error=0.01,count=100)


f.close()