
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

# skip warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[2]:


data= pd.read_csv("C:\\Users\\ushak\\Downloads\\Bank_Note_Authentication\\Bank_Note_Authentication\\BankNote_Authentication.csv")


# In[3]:


data.shape


# In[4]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# In[ ]:


data["class"]=data["class"].astype(str)


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# In[ ]:


num_attributes=list(data.select_dtypes(exclude='object').columns)
num_attributes


# In[ ]:


y.nunique()


# In[ ]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)


# In[ ]:


y_pred=dtc.predict(X_test)


# In[ ]:


score=accuracy_score(y_test,y_pred)

score


# In[ ]:


pickle_out = open("Bank_note.pkl","wb")
pickle.dump(dtc, pickle_out)
pickle_out.close()

dtc.predict([[0,1]])

