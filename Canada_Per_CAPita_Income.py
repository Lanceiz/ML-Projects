#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


# In[39]:


df=pd.read_csv('canada_per_capita_income.csv')


# In[40]:


df.head(3)


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('pci')
plt.scatter(df.year,df.pci)


# In[64]:


reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.pci )


# In[72]:


reg.predict([[2020]])


# In[73]:


reg.coef_


# In[ ]:




