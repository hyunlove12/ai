#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[4]:


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state = 1)


# In[13]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[7]:


print("scaler 적용 전 X_train :{}".format(X_train[:3]))


# In[11]:


# 무엇을 기준으로..?
print("scaler 적용 전 X_train의 max :{}".format(X_train[:3].max(axis=0)))
print("scaler 적용 전 X_train의 min :{}".format(X_train[:3].min(axis=0)))


# In[14]:


# 스케일러의 fit => 스케일 변경을 위한 데이터 인식 시켜 주기(데이터 입력)
#    transform => 입력된(fit) 데이터를 기반으로 데이터 스케일을 변경(데이터 변형)#
# fit_transform => 두 작업을 동시에 한다
scaler.fit(X_train) # 스케일을 변경할 데이터를 입력
X_train_scaled = scaler.transform(X_train) # 데이터 변경


# In[17]:


print("scaler 적용 후 X_train :{}".format(X_train_scaled[:3]))
# max값은 모두 1
# min값은 모두 0
print("scaler 적용 후 X_train의 max :{}".format(X_train_scaled.max(axis=0)))
print("scaler 적용 후 X_train의 min :{}".format(X_train_scaled.min(axis=0)))


# In[18]:


# X_test를 위한 슬케일러는 따로 안만드나?
# 단순하게 생각하면 스케일러에 대한 기준을 동일하게 가져간다?
# train데이터만 fit을 해야 한다. -> 데이터의 형상을 유지하기 위한 방법
# 원본 데이터의 분포도(scatter)를 유지하기 위한 방법
# 데이터 형상을 유지하면서 1과 0사이로 값을 줄이는것이 목적
X_test_scaled = scaler.transform(X_test)


# In[19]:


print("scaler 적용 후 X_test의 max :{}".format(X_test_scaled.max(axis=0)))
print("scaler 적용 후 X_test의 min :{}".format(X_test_scaled.min(axis=0)))


# In[20]:


# 스케일링 작업의 효과 확인하기
from sklearn.svm import SVC


# In[35]:


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state = 0)

svm = SVC(C=1000)
svm.fit(X_train, y_train)
print("훈련 세트의 정확도 : {:.2f}".format(svm.score(X_train, y_train)))
print("훈련 세트의 정확도 : {:.2f}".format(svm.score(X_test, y_test)))


# In[36]:


# MinMaxScaler 사용한 후 확인하기
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print("훈련 세트의 정확도 : {:.2f}".format(svm.score(X_train_scaled, y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(svm.score(X_test_scaled, y_test)))


# In[38]:


# 평균을 0으로, 분산을 1로
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print("스케일 조정된 테스트 세트의 정확도 : {:.2f}".format(svm.score(X_test_scaled, y_test)))


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




