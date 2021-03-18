#!/usr/bin/env python
# coding: utf-8

# INTERN NAME:- PRACHI PRADEEP PATIL

# # THE SPARKS FOUNDATION

# # "DATA SCIENCE AND BUSINESS ANALYTICS"

# TASK 01 - Prediction using supervised ML

# Statement :  To predict the percentage of an student based on the no. of study hours.What will be predicted score if a student                studies for 9.25 hrs/ day?

# Step 01 : Importing all required libraries.

# In[27]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# Step 02 : Importing the given data.

# In[28]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(25)


# Step 03 : Plotting the distribution of percentage of score and hours studied

# In[29]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# Thus the positive linear relation between percentage of score and hours studied is observed by above plotted graph.

# Step 04 : Preparing the data and dividing it into attributes and labels.

# In[ ]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# Step 05 : Splitting the data into training and testing models.

# In[31]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[32]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# Step 06 : Plotting the regression line.

# In[33]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# Step 07 : Making the predictions.

# In[34]:


# Testing data - In Hours
print(X_test)

# Predicting the scores
y_pred = regressor.predict(X_test) 


# In[35]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# Step 08 : Predicting the score if student studies for 9.25hrs/day.

# In[36]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# Step 09 : Evaluating the model to find the mean error.

# In[37]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# Thus the predicted score is 93.61 if a student studies for 9.25hrs/day.

# In[ ]:




