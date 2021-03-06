import sys 
f = open('./log.txt','w',encoding='utf-8')
sys.stdout = f

# import mainModule.LinearModel_2_Chapter.test_linear_model as test

import mainModule.util.test_util
import mainModule.LinearModel_3_Chapter.test_linear_model

f.close()