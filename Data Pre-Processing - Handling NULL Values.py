#!/usr/bin/env python
# coding: utf-8

# # Processing Null Values Before Training Your Model

# By Dishan Purkayastha

# In[4]:


import pandas as pd
import numpy as np


# In[22]:


df=pd.read_csv('C:/Users/devdi/Documents/DISHAN/Bengaluru_House_Data.csv')
df


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.tail()


# In[12]:


df.shape


# In[13]:


df.isnull()


# In[15]:


df.isnull().sum()


# In[16]:


df.isnull().sum().sum()


# In[18]:


# filling all null values with zero
df1=df.fillna(value=0)
df1.isnull().sum().sum()


# In[33]:


#filling NaN in different columns with different values:

temp=df.fillna({
    'price': 0.0,
    'balcony': df['balcony'].mean()
})

temp


# In[20]:


#filling null value with mean of column
df2=df.fillna(value=df['balcony'].mean())
df2.isnull().sum().sum()


# In[21]:


#completely dropping all the ROWs containing NaN
df3=df.dropna()
df3


# In[24]:


# drop using HOW - (any,all)
df4=df.dropna(how='all')
df4


# In[28]:


df5=df.dropna(how='any')
df5


# In[30]:


# Replacing NaN values:
import numpy as np
df6=df.replace(to_replace=np.nan,value=55)
df6


# In[31]:


#using interpolate( function)

#linear interpolation
df['balcony']=df['balcony'].interpolate(method='linear')


# In[32]:


df

