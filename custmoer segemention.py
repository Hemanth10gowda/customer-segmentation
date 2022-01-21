#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df=pd.read_csv(r'C:\Users\BHOOMI\Downloads\Mall_Customers.csv')
df.head()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


df.dtypes


# In[13]:


df.isnull().sum()


# In[14]:


df.drop(['CustomerID'],axis=1, inplace=True)


# In[15]:


df.head()


# In[16]:


plt.figure(1, figsize=(15,6))
n=0
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace =0.5, wspace = 0.5)
    sns.distplot(df[x], bins=20)
    plt.title ('Distplot of {}'.format(x))
plt.show()


# In[17]:


plt.figure(figsize=(15,5))
sns.countplot(y='Gender',data=df)
plt.show()


# In[18]:


plt.figure(1, figsize=(15,8))
n=0
for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace =0.5, wspace = 0.5)
    sns.violinplot(x = cols,y= 'Gender',data=df)
    plt.ylabel('Gender' if n==1 else '')
    plt.title ('Distplot of {}'.format(x))
plt.show()


# In[19]:


age_18_25 = df.Age[(df.Age>=18) & (df.Age<=25)]
age_26_35 = df.Age[(df.Age>=26) & (df.Age<=35)]
age_36_45 = df.Age[(df.Age>=36) & (df.Age<=45)]
age_46_55 = df.Age[(df.Age>=46) & (df.Age<=55)]
age_55above = df.Age[df.Age>=56]

agex = ["18-25","26-35","36-45","46-55","55+"]
agey =[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=agex, y=agey)
plt.title("Number of customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of customer")
plt.show()


# In[21]:


X1=df.loc[:, ["Age","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker="8")
plt.xlabel("k value")
plt.ylabel("wcss")
plt.show()


# In[22]:


kmeans = KMeans(n_clusters=4)

label = kmeans.fit_predict(X1)

print(label)


# In[23]:


print(kmeans.cluster_centers_)


# In[24]:


plt.scatter(X1[:,0], X1[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.title('clusters of customers')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[25]:


X2=df.loc[:, ["Annual Income (k$)","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker="8")
plt.xlabel("k value")
plt.ylabel("wcss")
plt.show()


# In[26]:


kmeans = KMeans(n_clusters=5)

label = kmeans.fit_predict(X2)

print(label)


# In[27]:


print(kmeans.cluster_centers_)


# In[28]:


plt.scatter(X2[:,0], X2[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.title('clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[30]:


X3=df.iloc[:,1:]


wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker="8")
plt.xlabel("k value")
plt.ylabel("wcss")
plt.show()


# In[31]:


kmeans = KMeans(n_clusters=5)

label = kmeans.fit_predict(X3)

print(label)


# In[32]:


print(kmeans.cluster_centers_)


# In[33]:


clusters = kmeans.fit_predict(X3)
df["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0],df["Annual Income (k$)"][df.label == 0],df["Spending Score (1-100)"][df.label == 0], c= 'blue', s=60)
ax.scatter(df.Age[df.label == 1],df["Annual Income (k$)"][df.label == 1],df["Spending Score (1-100)"][df.label == 1], c= 'red', s=60)
ax.scatter(df.Age[df.label == 2],df["Annual Income (k$)"][df.label == 2],df["Spending Score (1-100)"][df.label == 2], c= 'green', s=60)
ax.scatter(df.Age[df.label == 3],df["Annual Income (k$)"][df.label == 3],df["Spending Score (1-100)"][df.label == 3], c= 'orange', s=60)
ax.scatter(df.Age[df.label == 4],df["Annual Income (k$)"][df.label == 4],df["Spending Score (1-100)"][df.label == 4], c= 'purple', s=60)
ax.view_init(30,185)

plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")

plt.show()


# In[ ]:




