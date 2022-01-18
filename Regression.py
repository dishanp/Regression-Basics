#!/usr/bin/env python
# coding: utf-8

# # Regression Basics
# ### By Dishan Purkayastha

# In[68]:


import pandas as pd;
import numpy as numpy;
from scipy import stats;
import matplotlib.pyplot as plt;
from sklearn.metrics import r2_score;
from sklearn import linear_model;


# Mean, Median, Mode, Variance & Standard Deviance

# In[7]:


data = [32,111,138,28,59,77,97]
x=numpy.mean(data);
print(x);


# In[8]:


data = [32,111,138,28,59,77,97]
x=numpy.median(data);
print(x);


# In[12]:


data = [32,111,138,28,59,77,97,32,3]
x=stats.mode(data);
print(x);


# In[18]:


data = [32,111,138,28,59,77,97]
x=numpy.var(data);
print(x);


# In[17]:


data = [32,111,138,28,59,77,97]
x=numpy.std(data);
print(x);


# Generating Random Datasets & Corresponding Plots

# In[24]:


#Uniform Distribution between 0 and 10 with 30 data points

x=numpy.random.uniform(0,10,30)
print(x)


# In[28]:


#Plotting the uniform data

plt.hist(x,10);
plt.show();


# In[32]:


#Normal Distribution between 0 and 10 with 100 data points

x=numpy.random.normal(0,10,100)
print(x)


# In[38]:


#Plotting the normal data

plt.hist(x,10);
plt.show();


# In[39]:


#Scatter Plot

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x,y)


# In[43]:


# Random Scatter Plot for bigger data set

x=numpy.random.uniform(0,10,300);
y=numpy.random.uniform(50,100,300);
plt.scatter(x,y);


# Linear Regression

# In[46]:


x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope,intercept,r,p,std_err = stats.linregress(x,y)

def myfunc(x):
    return slope*x + intercept

mymodel=list(map(myfunc,x))

plt.scatter(x,y);
plt.plot(x,mymodel);
plt.show()


# Analysis Of r-value

# In[47]:


print(r)


# Future Predictions:

# In[51]:


res=myfunc(10)
print(res)

res=myfunc(20)
print(res)

res=myfunc(40)
print(res)


# Polynomial Regression:

# In[52]:


x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

plt.scatter(x,y)


# In[53]:


mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


# r2 Value Analysis:

# In[56]:


print(r2_score(y, mymodel(x)));


# Predicting Future Values:

# In[58]:


res=mymodel(10)
print(res)

res=mymodel(20)
print(res)

res=mymodel(30)
print(res)


# Mulltiple Regression:

# In[61]:


#importing a new dataset 'Cars':

df=pd.read_csv('C:\\Users\\devdi\\Documents\\DISHAN\\cars.csv')


# In[62]:


df.head()


# In[63]:


df.tail(10)


# In[67]:


x=df[['Weight','Volume']]
y=df['CO2']


# In[69]:


regr = linear_model.LinearRegression()
regr.fit(x,y);


# Predicting a future value based on 2 value inputs:

# In[71]:


res=regr.predict([[2300,1300]])
print(res)


# Checking Coefficient Value:

# In[72]:


print(regr.coef_)


# Train-Test Split On Datasets:

# In[75]:


x=df[['Weight']]
y=df['CO2']
plt.scatter(x,y);
plt.show();


# In[78]:


#splitting data set into 80(Train):20(Test)
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

plt.scatter(train_x,train_y);
plt.show();


# In[85]:


# building a polynomial regression model on the train data-set:

import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))

print(r2)


# In[86]:


# Now checking r2 score using test dataset:

r2 = r2_score(test_y, mymodel(test_x))

print(r2)

