#!/usr/bin/env python
# coding: utf-8

# In[1]:


# seaborn
# pip install seaborn
import matplotlib.pyplot as plt # seaborn을 사용 할 때 matplotlib도 필수적으로 import
import numpy as np
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set_style('whitegrid') # document의 스타일에 관련된 내용이 있음 -> 한글등을 불러온 작업을 다시 해야한다. 
# seaboarn이 자동으로 부럴와진다?


# In[4]:


x = np.linspace(0, 14, 100) # 0부터 14까지 100개 arange는 간격을 지정 linspace는 개수를 지정

y1 = np.sin(x)
y2 = 2 * np.sin(x + 0.5)
y3 = 3 * np.sin(x + 1.0)
y4 = 4 * np.sin(x + 1.5)


# In[8]:


plt.figure(figsize=(10, 6))
plt.plot(x, y1, x, y2, x, y3, x, y4)
plt.show()


# In[14]:


tips = sns.load_dataset('tips') # 레스토랑 데이터
tips.head()


# In[23]:


# seabonr은 x축, y축 컬럼 지정을 자동으로 판단하게 해줄 수 있다.
plt.figure(figsize=(10, 6))
# boxplot -> 표준편차, 분산 등까지 보여준다 -> 통계적인 의미를 보여준다
# 위의 점들은 이상치, 박스안에 있는 것이 평균치, 박스안의 선은 전체 평균 
# hue -> 기준컬럼을 구분하여 통계를 보여준다
# 연속적특성 -> 실수 
# 범주형특성 -> 정수, 문자형 -> 카테고리
# hue는 범주형 특성을 지정해야 한다. 
# sns.boxplot(x='day', y='total_bill', hue='smoker', data=tips) # 판다스 데이터 프레임을 파라미터로 하여 x축컬럼이름, y축 컬럼이름
sns.boxplot(x='day', y='total_bill', palette='Set3', hue='sex', data=tips) # 판다스 데이터 프레임을 파라미터로 하여 x축컬럼이름, y축 컬럼이름
plt.show()


# In[27]:


# 상관관계를 파악하기 좋은 lmplot
# lmplot -> 상관도 계산
sns.set_style('darkgrid')
# hue는 범주형 특성
sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker') # , height=10)
plt.show()


# In[33]:


flights = sns.load_dataset('flights')
flights.head()


# In[34]:


# 1개열을 컬럼으로 보낸다
# pivot
flights = flights.pivot('month', 'year', 'passengers') # 인덱스, 컬럼, 데이터
flights.head()


# In[40]:


# heatmap
# 하이퍼 파라미터에 다른 정확도를 볼 수 있다.
plt.figure(figsize=(10, 8))
sns.heatmap(flights, annot=True, fmt='d') # anot : 맴에 숫자 표시, fmt : 포현 형식, d : 정수형태, 기본은 실수형태(f)로 표현 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




