#!/usr/bin/env python
# coding: utf-8

# In[1]:


#필요 라이브러리 임포트
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


# In[2]:


mglearn.plots.plot_linear_regression_wave() # mglearn 샘플 그래프 보기


# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[7]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[9]:


# _붙은 변수는 모델이 학습하여 만들어낸 변수
# n개의 특성에 대한 값들이 나와야 하기 때문에 리스트로 출력 된다
print("linreg의 계수(COFFICIENT, WEIGHT) : {}".format(linreg.coef_))
print("linreg의 편향(INTERCEPT, BIAS) : {}".format(linreg.intercept_))


# In[10]:


print("훈련 세트에 대한 점수 : {:.2f}".format(linreg.score(X_train, y_train)))
print("테스트 세트에 대한 점수 : {:.2f}".format(linreg.score(X_test, y_test)))

# 훈련 세트에 대한 점수 : 0.67
# 테스트 세트에 대한 점수 : 0.66
# 과소 적합의 상태 -> 단순한 모델
# 선형회귀는 저차원 모델일 수록 불리하다 
# 수개의 특성이 있을 때 예측도가 높아진다


# In[ ]:


# 회귀에서의 점수는 R^2(r제곱)


# In[32]:


# feature(컬럼)이 많은 보스턴 데이터 세트로 선형회귀 확인
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
boston_lin_reg = LinearRegression().fit(X_train, y_train)


# In[18]:


print("boston_lin_reg의 계수(COFFICIENT, WEIGHT) : {}".format(boston_lin_reg.coef_))
print("boston_lin_reg의 편향(INTERCEPT, BIAS) : {}".format(boston_lin_reg.intercept_))


# In[ ]:


print("훈련 세트 점수 : {:.2f}".format(boston_lin_reg.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(boston_lin_reg.score(X_test, y_test)))


# In[ ]:


# 훈련 세트 점수 : 0.95
# 테스트 세트 점수 : 0.61
# 과대적합의 상태 -> 훈련 세트에만 적합한 모델이다  
# 데이터를 추가하거나 복잡도를 수정해야 한다 # 가중치를 낮춰주는 것이 좋다


# In[20]:


from sklearn.linear_model import Ridge
# 특정 특성을 0으로 만들지는 않는다
ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수 : {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(ridge.score(X_test, y_test)))


# In[ ]:


# a(alpha) -> 가중치에 대한 패널티
# a(alpha)값은 보통 로그 스케일(100,10, 1, 0.1, 0.01, ... ,)으로 부여하는 것이 일반적
# a(alpha)의 기본갑은 1.0


# In[25]:


# ridge의 alpha값을 부여
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수 : {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(ridge10.score(X_test, y_test)))


# In[26]:


# ridge에 낮은 alpha값을 부여
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수 : {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(ridge01.score(X_test, y_test)))


# In[28]:


plt.plot(ridge10.coef_, '^', label="Ridge alpha 10")
plt.plot(ridge.coef_, 's', label="Ridge alpha 1.0")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha 0.1")

plt.plot(boston_lin_reg.coef_, 'o', label="LinearRegression")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.hlines(0, 0, len(boston_lin_reg.coef_))
plt.ylim(-25, 25)
plt.legend()


# In[34]:


# Lasso
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)


print("훈련 세트 점수 : {:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(lasso.score(X_test, y_test)))


# In[45]:


# Ridge = 경사하강법 (미분을 통해 오차가 최소화 되는 기울기를 찾음)
# Lasso = 좌표하강법 (특성 하나의 오차에 대해 좌표축을 따라 오차가 최소회 되는 곳을 찾음)
# 즉 좌표하강법은 학습 과정이 여러번 진행 되어야 하기 때문에 max_iter값을 지정하여 
# 최소화된 오차를 계속 개선할 수 있도록 해야한다. 
# alpha를 줄이게 되면 가장 낮은 오차를 찾아가는 반복횟수가 늘어난다.  -> 최대반복횟수르 지정한다.

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수 : {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(lasso001.score(X_test, y_test)))


# In[47]:


print("사용된 특성의 수 : {}".format(np.sum(lasso001.coef_ != 0))) # true값은 1


# In[50]:


# 가중치를 높이면 선택되는 특성이 줄어든다
lasso0001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train) # 가중치를 낮췄기 때문에 특성이 더 활성화 된다. -> 과대적합
print("훈련 세트 점수 : {:.2f}".format(lasso0001.score(X_train, y_train)))
print("테스트 세트 점수 : {:.2f}".format(lasso0001.score(X_test, y_test)))
print("사용된 특성의 수 : {}".format(np.sum(lasso0001.coef_ != 0))) # true값은 1


# In[51]:


plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha = 0.01")
plt.plot(lasso0001.coef_, 'v', label="Lasso alpha = 0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha = 0.1") # 특성값이 절대 0이 되지 않는다. 
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




