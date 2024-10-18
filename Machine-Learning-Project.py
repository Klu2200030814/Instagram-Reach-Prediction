#!/usr/bin/env python
# coding: utf-8

# In[275]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[276]:


data = pd.read_csv("Instagram_data.csv")


# In[277]:


print(data.head())


# In[278]:


data.isnull().sum()


# In[279]:


data = data.dropna()


# In[280]:


data.info()


# In[281]:


data = data.fillna(0)


# In[282]:


numeric_columns = ['Post Reach', 'Likes']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Likes'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Likes')
plt.xlabel('Post Reach')
plt.ylabel('Likes')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[283]:


numeric_columns = ['Post Reach', 'Comments']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Comments'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Comments')
plt.xlabel('Post Reach')
plt.ylabel('Comments')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[284]:


numeric_columns = ['Post Reach', 'Saves']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Saves'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Saves')
plt.xlabel('Post Reach')
plt.ylabel('Saves')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[285]:


numeric_columns = ['Post Reach', 'Shares']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Shares'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Shares')
plt.xlabel('Post Reach')
plt.ylabel('Shares')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[286]:


numeric_columns = ['Post Reach', 'Profile Visits']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Profile Visits'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Profile Visits')
plt.xlabel('Post Reach')
plt.ylabel('Profile Visits')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[287]:


numeric_columns = ['Post Reach', 'Follows']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Follows'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Follows')
plt.xlabel('Post Reach')
plt.ylabel('Follows')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[288]:


numeric_columns = ['Post Reach', 'Caption Length']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Caption Length'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Caption Length')
plt.xlabel('Post Reach')
plt.ylabel('Caption Length')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[289]:


numeric_columns = ['Post Reach', 'Hashtags']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in 'Post Reach' or 'Likes' after conversion
data = data.dropna(subset=numeric_columns)

# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Post Reach'], data['Hashtags'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Post Reach vs Hashtags')
plt.xlabel('Post Reach')
plt.ylabel('Hashtags')
plt.grid(True)
plt.xscale('log')  # Log scale can help visualize larger ranges
plt.yscale('log')  # Log scale for likes if needed
plt.show()


# In[296]:


# Convert 'Type of Post' to numeric
data['Type of Post'] = data['Type of Post'].replace({'Image': 0, 'Reel': 1})

# Features to use for training
X = np.array(data[['Post Reach','Profile Visits', 'Follows', 'Hashtags', 'Caption Length', 'Type of Post']])



# Predict Likes
y_likes = np.array(data['Likes'])
X_train_likes, X_test_likes, y_train_likes, y_test_likes = train_test_split(X, y_likes, test_size=0.2, random_state=42)


# Predict Shares
y_shares = np.array(data['Shares'])
X_train_shares, X_test_shares, y_train_shares, y_test_shares = train_test_split(X, y_shares, test_size=0.2, random_state=42)


# Predict Comments
y_comments = np.array(data['Comments'])
X_train_comments, X_test_comments, y_train_comments, y_test_comments = train_test_split(X, y_comments, test_size=0.2, random_state=42)


# Predict Saves
y_saves = np.array(data['Saves'])
X_train_saves, X_test_saves, y_train_saves, y_test_saves = train_test_split(X, y_saves, test_size=0.2, random_state=42)


# In[297]:


model_likes = PassiveAggressiveRegressor()
model_likes.fit(X_train_likes, y_train_likes)
model_likes.score(X_test_likes, y_test_likes)
model_shares = PassiveAggressiveRegressor()
model_shares.fit(X_train_shares, y_train_shares)
model_shares.score(X_test_shares,y_test_shares)
model_comments = PassiveAggressiveRegressor()
model_comments.fit(X_train_comments, y_train_comments)
model_comments.score(X_test_comments,y_test_comments)
model_saves = PassiveAggressiveRegressor()
model_saves.fit(X_train_saves, y_train_saves)
model_saves.score(X_test_saves, y_test_saves)


# In[298]:


# Example feature input for prediction
features = np.array([[50000,5400, 10000, 4, 120, 0]])  # Example input (Post Reach, Profile Visits, Follows, Hashtags, Caption Length, Type of Post)

# Making predictions
predicted_likes = model_likes.predict(features)
predicted_shares = model_shares.predict(features)
predicted_comments = model_comments.predict(features)
predicted_saves = model_saves.predict(features)

print(f'Predicted Likes: {predicted_likes[0]}')
print(f'Predicted Shares: {predicted_shares[0]}')
print(f'Predicted Comments: {predicted_comments[0]}')
print(f'Predicted Saves: {predicted_saves[0]}')


# In[299]:


# For Likes
predicted_likes = model_likes.predict(X_test_likes)
mse_likes = mean_squared_error(y_test_likes, predicted_likes)
r2_likes = r2_score(y_test_likes, predicted_likes)

# For Shares
predicted_shares = model_shares.predict(X_test_shares)
mse_shares = mean_squared_error(y_test_shares, predicted_shares)
r2_shares = r2_score(y_test_shares, predicted_shares)

# For Comments
predicted_comments = model_comments.predict(X_test_comments)
mse_comments = mean_squared_error(y_test_comments, predicted_comments)
r2_comments = r2_score(y_test_comments, predicted_comments)

# For Saves
predicted_saves = model_saves.predict(X_test_saves)
mse_saves = mean_squared_error(y_test_saves, predicted_saves)
r2_saves = r2_score(y_test_saves, predicted_saves)


# In[300]:


# After making predictions
print(f'Shape of y_test_likes: {y_test_likes.shape}, predicted_likes: {predicted_likes.shape}')
print(f'Shape of y_test_shares: {y_test_shares.shape}, predicted_shares: {predicted_shares.shape}')
print(f'Shape of y_test_comments: {y_test_comments.shape}, predicted_comments: {predicted_comments.shape}')
print(f'Shape of y_test_saves: {y_test_saves.shape}, predicted_saves: {predicted_saves.shape}')


# In[301]:


print(f'Likes - Mean Squared Error: {mse_likes}, R² Score: {r2_likes}')
print(f'Shares - Mean Squared Error: {mse_shares}, R² Score: {r2_shares}')
print(f'Comments - Mean Squared Error: {mse_comments}, R² Score: {r2_comments}')
print(f'Saves - Mean Squared Error: {mse_saves}, R² Score: {r2_saves}')


# In[ ]:




