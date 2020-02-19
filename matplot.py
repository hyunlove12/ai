#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
# matplotlib을 주피터 노트북에 표현하는 키워드
get_ipython().run_line_magic('matplotlib', 'inline')
# 한글 사용시 폰트변경 필요


# In[5]:


plt.figure
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1]) # 점을 선으로 연결해준다
plt.show()


# In[10]:


t = np.arange(0, 12, 0.01) # 0부터 12까지 0.01간격으로(linspace와 비슷)
y = np.sin(t) # t에 대한 싸인 값 # 브로드 캐스팅 적용


# In[15]:


plt.figure(figsize=(10, 6)) # figure 그래프를 그릴 공간의 크기
plt.plot(t, y) # x축을 지정하지 않으면 넣은 값은 y축으로 가고 x축은 자도응로 계산
plt.show()


# In[25]:


# figure를 호출 할때마다 matplotlib의 영역이 초기화 된다.
plt.figure(figsize=(20, 6))

# 하나의 figure에 여러개의 그래프를 그릴 수 있다.
plt.plot(t, y)
plt.plot(t, np.cos(t))

plt.grid() # 그래프의 그리드 그리기(격자무늬)
plt.xlabel('time') # x축 제목
plt.ylabel('Amplitude') # y축 제목
plt.title('Example Of Sinewave') # 그래프 전체 제목

plt.show()


# In[40]:


# figure를 호출 할때마다 matplotlib의 영역이 초기화 된다.
plt.figure(figsize=(20, 6))

# 하나의 figure에 여러개의 그래프를 그릴 수 있다.
plt.plot(t, y, lw=7, label='sin', color='g', linestyle='dotted') # lw -> 선 굵기
plt.plot(t, np.cos(t), label='cos', linestyle='dashed')
plt.legend(loc='best') # 범례 -> label표시 / loc=best-> 최적의 장소를 찾아준다

plt.grid() # 그래프의 그리드 그리기(격자무늬)
plt.xlabel('time') # x축 제목
plt.ylabel('Amplitude') # y축 제목
plt.title('Example Of Sinewave') # 그래프 전체 제목

plt.show()


# In[51]:


t = [0,1,2,3,4,5,6]
y = [2,5,3,7,1,9,3]
plt.figure(figsize=(10, 6))
plt.plot(t, y, marker='*', markerfacecolor='r', markersize=20) # o, ^, *, v, .... # https://matplotlib.org/3.1.1/api/markers_api.html
plt.show()


# In[57]:


# plot -> 추세를 확인할 때(증가량, 감소량 등등)
# box -> 어느정도의 양이 있구나 확인 할때 
# scatter -> 분포도 확인할때 

colormap = y # y의 값에 따라 색상을 그라데이션 형식으로 표현

plt.figure(figsize=(10, 6))
plt.scatter(t, y, marker='*', c=colormap) # 산점도 
plt.colorbar()
plt.show()


# In[ ]:


# matplotlib 그래프 모양 
# https://matplotlib.org/3.1.1/gallery/index.html


# In[ ]:


# 히스토그램 -> 빈도


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




