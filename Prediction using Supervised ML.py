#!/usr/bin/env python
# coding: utf-8

# # 1st Step is to Make sure That you Import necessary libraries

# ## Done By Ahmed Hesham Agamy Abdelrasol

# In[2]:


pip install pandas numpy matplotlib scikit-learn


# ## Write down your code

# In[6]:


## Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## Load the dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

## Display the first few rows of the dataset
print(data.head())

## Plot the data to visualize the relationship between study hours and scores
plt.scatter(data['Hours'], data['Scores'])
plt.title('Study Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

## Prepare the data for training
X = data.iloc[:, :-1].values  # Features (Hours)
y = data.iloc[:, 1].values    # Labels (Scores)

## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Initialize the linear regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

## Make predictions on the test set
y_pred = model.predict(X_test)

## Visualize the regression line
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.title('Study Hours vs Scores with Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

## Predict the percentage for a student who studies 9.25 hours/day
hours = np.array([[9.25]])
predicted_score = model.predict(hours.reshape(-1, 1))[0]


print(f'Predicted Score for 9.25 hours/day of study: {predicted_score:.2f}%')

print("Done By Ahmed Hesham Agamy Abdelrasol")


# In[ ]:




