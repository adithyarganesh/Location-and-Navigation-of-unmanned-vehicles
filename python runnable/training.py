
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.utils import shuffle


# In[2]:

from sklearn.model_selection import train_test_split


# # In[3]:

# data = pd.read_csv('900.340.txt',header = None)
# data.head()


# # In[4]:

# data[0] = data[0]/340*330


# # In[7]:

# #df.columns = [str(x) for x in df.columns]
# data.head()


# # In[8]:

# data.to_csv('900.txt', index=False)


# # In[9]:

# full_data = data
# full_data.drop(full_data.index, inplace=True)


# # In[ ]:




# # In[10]:

# full_data.columns = [str(x) for x in full_data.columns]
# full_data.drop(['2','3'],axis=1)


# # In[ ]:




# # In[11]:

# data_distance = []
# i = 0
# while i < 1500:
#     i = i+150
#     data_distance.append(i)


# # In[12]:

# data_distance


# # In[14]:

# for i in data_distance:
#     file_name = str(i) + ".txt"
#     df = pd.read_csv(file_name, header = None)
#     df_new = df
#     df_new[0] = df[0].round(2)
#     df_new[1] = i
#     df_new.columns = [str(x) for x in df_new.columns]
#     df_new = df_new.drop(['2','3'],axis=1)
#     df_new.drop(df.index[:50], inplace=True)
#     df_new = df_new.drop_duplicates()
#     frames = [df_new, full_data]
#     full_data = pd.concat(frames)
#     full_data = full_data.drop_duplicates()
    


# # In[ ]:




# # In[15]:

# full_data = full_data.drop(['2','3'],axis=1)
# full_data.info()


# # In[16]:

# full_data.head()


# # In[368]:

# full_data.to_csv('distance_dataset.txt', index = False)


# # In[ ]:




# # In[ ]:




# # In[ ]:




# # In[ ]:




# # In[ ]:




# # In[ ]:




# In[17]:

dataset = pd.read_csv('distance_dataset.txt',header = None)


# In[18]:

dataset.drop(dataset.index[:1], inplace=True)


# In[19]:

# dataset.info()


# # In[20]:

# dataset.head()


# In[21]:

dataset = shuffle(dataset)


# In[22]:

dataset.columns = [str(x) for x in dataset.columns]


# In[ ]:




# In[23]:

import matplotlib.pyplot as plt


# In[24]:

plt.scatter(dataset['0'],dataset['1'])
plt.show()


# In[ ]:




# In[ ]:




# In[25]:

X = dataset
X = X.drop(['1'], axis=1)


# In[26]:

y = dataset['1']


# In[27]:

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# In[28]:

#Now we have split the training and test data. We now perform various machine learning algorithms to predic an accurate result
#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor


# In[29]:

#Linear Regresion
linear = LinearRegression()

linear.fit(X_train,y_train)

Y_pred = linear.predict(X_test)

linear.score(X_train,y_train)


# In[31]:

import pickle


# In[34]:


# filename = 'finalized_model.sav'
# pickle.dump(linear, open(filename, 'wb'))
 
# # some time later...
# result = loaded_model.score(X_test, y_test)
# print(result)
# # load the model from disk

# def prediction(value):
#     loaded_model = pickle.load(open(filename, 'rb'))
#     return loaded_model.predict(value)


# In[382]:

#logistic regression
# logreg = LogisticRegression()

# logreg.fit(X_train,y_train)

# Y_pred = logreg.predict(X_test)

# logreg.score(X_train,y_train)


# In[383]:


#Support Vector Machines

# svc = SVC()

# svc.fit(X_train, y_train)

# Y_pred = svc.predict(X_test)

# svc.score(X_train, y_train)



# In[384]:

# # Random Forests

# random_forest = RandomForestClassifier(n_estimators=100)

# random_forest.fit(X_train, y_train)

# Y_pred = random_forest.predict(X_test)

# random_forest.score(X_train, y_train)


# In[385]:

# #knn classifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, y_train)


# In[386]:

regressor = RandomForestRegressor(n_estimators=150, random_state = 42)

regressor.fit(X_train, y_train)

Y_pred = regressor.predict(X_test)

regressor.score(X_train, y_train)


# In[ ]:




# In[387]:

#predicted an unknown range value to maximum accuracy
# dataframe = pd.DataFrame({
#         "input_range": test_df["input_range"],
#         "predicted": Y_pred
#         })
# dataframe.to_csv('ips.csv', index=False)


# In[393]:

regressor.predict(1000)


# In[415]:

print(linear.predict(144))
print(linear.predict(351))
print(linear.predict(500))
print(linear.predict(788))
print(linear.predict(921))
print(linear.predict(1139))


# In[410]:

from sklearn.metrics import mean_squared_error, r2_score


# In[418]:

mean_df = []
for i in data_distance:
    file_name = str(i) + ".txt"
    df = pd.read_csv(file_name, header = None)
    df.drop(df.index[:50], inplace=True)
    df_new = df
    df_new[0] = df[0].round(2)
    df_new[1] = i
    df_new.columns = [str(x) for x in df_new.columns]
    print("Mean squared error: %.2f"% mean_squared_error(df_new['0'], df_new['1']))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(df_new['0'], df_new['1']))
    mean_df.append(df['0'].mean())
mean_df


# In[409]:

# Plot outputs
plt.scatter(mean_df, data_distance,  color='black')
plt.plot(mean_df, data_distance, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[411]:

print("Mean squared error: %.2f"% mean_squared_error(dataset['0'], dataset['1']))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(dataset['0'], dataset['1']))


# In[36]:

def main():
    print(prediction(166))


# In[37]:

if __name__ == "__main__":
    main()

