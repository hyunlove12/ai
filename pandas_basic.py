#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[7]:


# Series는 판다스의 제일 기본적인 자료형이다.
# 즉, 데이터프레임은 시리즈의 집합이다. 

s = pd.Series([1, 2, 3, np.nan, 6, 7])
print(type(s))
s


# In[20]:


# periods -> 기간설정 5일치의 날자를 만든다
dates = pd.date_range('20200219', periods=5)
dates


# In[ ]:


# 데이터 프레임을 만들기 -> dict, 리스트, 시리즈, 컬럼과 데이터를 구분하여 생성 


# In[22]:


df = pd.DataFrame(np.random.randn(5, 4),  # 5행 4열 형태의 난수 발생
                  index = dates, # 위에서 만든 날짜를 인덱스로 사용 # 인덱스는 유니크하다. # 행의 개수와 인덱스의 개수는 언제나 일치해야
                  columns=['A', 'B', 'C', 'D']
                ) 
df.head() # head 데이터를 앞에서부터 출력(기본은 5개)


# In[23]:


df.head(3)


# In[26]:


df.tail(2) # 뒤에서부터 출력(기본은 5개)


# In[31]:


# 데이터 프레임 정보
# 인덱스 정보, 컬럼 정보, 값 정보, 데이터 프레임 전체 정보
df.index # 리스트로 반환
# df.index[0]

df.columns # 컬럼 정보 object타입은 문자열 # astype이용하여 형변환 가능


# In[32]:


df.values # 실제 데이터 -> 2차원 배열


# In[33]:


df.info() # 데이터 종합 정보


# In[40]:


# 데이터 프레임 통계 -> 새로운 데이터 프레임으로 반환
# 숫자데이터를 대상으로 한다.
# std -> 표준편차
# 분위 지점
# iqr지점
df.describe()


# In[46]:


# 통계에서 중요한 것 중에 하나는 정렬
# 정렬을 하기 위해 필요한 것은 기준

df.sort_values(by='B', ascending=False) # 컬럼 B기준으로 내림차순 정렬


# In[45]:


df['A'] # 컬럼 선택 # 딕셔너리형과 비슷하기 때문


# In[47]:


feature_name = ['A', 'D']
df[feature_name] # 여러개 컬럼을 선택할 경우 리스트를 넘겨야 한다.


# In[48]:


# 슬라이싱 기법은 행기준으로 적용된다
df[0:2] # 0~1번째 행 (이상 미만)


# In[49]:


df['20200202' : '20200224'] # 날짜로 슬라이싱도 가능


# In[60]:


# 특정 위치의 데이터 확인하기  -> loc(location)
# loc에는 두가지 매개변수가 존재함-> loc[행, 열] -> 열이 생략되면 행만 선택
df.loc['20200222']
df.loc['20200222', 'A']
df.loc['20200222': '20200223']
df.loc['20200222': '20200223', :]
df.loc['20200222', ['A', 'B']]


# In[64]:


df.loc['20200219': '20200221', ['A', 'D']] # loc는 [이상, 이하]


# In[68]:


df.loc[:, ['A', 'C']] # loc는 [이상, 이하]


# In[72]:


df.iloc[3]
df.iloc[3 : 4] # iloc -> 숫자로만 슬라이싱 가능


# In[73]:


df.iloc[2:5, 0:2]


# In[74]:


df.A


# In[78]:


# 조건으로 데이터 선택하기
# mask만들기
# mask = df.A + 10
# 넘파이 브로드 캐스팅 기법 -> 0차원을 1차원 형태로 바꿔준다
mask = df.A > 0
mask


# In[81]:


df[mask] # 조건에 충족하는 값만 출력된다. 


# In[83]:


mask = df > 0 # 전체 데이터 기준이 된다. 0차원 -> 2차원 브로드캐스팅
mask


# In[84]:


df[mask] # 매칭이 안되는 부분은 모두 NaN로 보여진다


# In[87]:


# 데이터 프레임 복사하기 
df2 = df # 얕은 복사 -> 메모리에는 1개의 데이터프레임 만 존재한다
df3 = df.copy() # 깊은 복사
df4 = df[:] # 깊은 복사 -> copy와 같은 기능


# In[110]:


# 판다스에서 밑의 코드는 열의 추가와 수정 모두를 담당한다.
df3['E'] = ['one', 'one', 'two', 'three', 'four']
df3


# In[105]:


# 데이터의 존재 유무 판단하기 -> isin
mask = df3['E'].isin(['two', 'four']) # mask만들기와 비슷


# In[106]:


df3[mask]


# In[104]:


mask = df3['E'].isin(['one', 'four'])
df3[mask][['A', 'C']]


# In[108]:


mask = df3['E'].isin(['one', 'four'])
df3.loc[mask, ['A', 'D']] # loc의 핵심은 행에대한 필터링


# In[111]:


# 데이터의 정합성
# 한 셀에 데이터가 여러개 등등 의 문제 


# In[113]:


df3['A'] = np.nan
df3


# In[114]:


df3.isnull() # NaN값 확인


# In[115]:


df3['A'].isnull()


# In[116]:


df3['B'].isnull()


# In[119]:


df3[df3['A'].isnull()] # A컬럼 중 nan이 값


# In[122]:


df_test = df[df > 0]


# In[123]:


df_test


# In[135]:


df_test.loc[df_test['A'].isnull(), 'A'] = 0  # A커럶의 nan값을 0으로 -> 행 선택 후 열 선택
df_test


# In[136]:


df_test.notnull() # 값이 있는것만 true


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





# In[ ]:





# In[ ]:




