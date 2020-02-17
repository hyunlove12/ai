#!/usr/bin/env python
# coding: utf-8

# In[3]:


#사용 할 라이브리 임포트하기

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
plt.rcParams['axes.unicode_minus'] = False


import platform
path = 'c:/Windows/Fonts/malgun.ttf'
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~~')


# In[ ]:





# In[8]:


# 1. forge 데이터 셋
X, y = mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['클래스 0', '클래스 1'], loc=4)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[13]:


# 회귀 예제 데이터 셋 wave
X, y = mglearn.datasets.make_wave(n_samples=40) # 샘플의 개수를 지정함

plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('특성')
plt.ylabel('타겟')
plt.show()


# In[ ]:




