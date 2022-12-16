#!/usr/bin/env python
# coding: utf-8

# ### Mohamed Muad Mafaz
# ### COADDS 202P-009

# ## Data Analysis

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib notebook


# In[2]:


#Loading the data set
vehicle_data = pd.read_csv(r"C:\Users\acer\Downloads\vehicle_data.csv", low_memory=False)
vehicle_data.head()


# In[3]:


vehicle_data.shape


# In[4]:


vehicle_data.dtypes


# In[5]:


vehicle_data.isnull().sum()


# In[6]:


#Filling Null values with mode as it has a large number of records
vehicle_data['Edition'].fillna(vehicle_data['Edition'].mode()[0], inplace = True)
vehicle_data['Body'].fillna(vehicle_data['Body'].mode()[0], inplace = True)
vehicle_data['Capacity'].fillna(vehicle_data['Capacity'].mode()[0], inplace = True)
vehicle_data['Description'].fillna(vehicle_data['Description'].mode()[0], inplace = True)
vehicle_data['Transmission'].fillna(vehicle_data['Transmission'].mode()[0], inplace = True)

#Dropping Null values as it has only a fewer number records which will not make any difference
vehicle_data = vehicle_data[vehicle_data['Model'].notna()]
vehicle_data = vehicle_data[vehicle_data['Transmission'].notna()]
vehicle_data = vehicle_data[vehicle_data['Seller_name'].notna()]


# In[7]:


vehicle_data.isnull().sum()


# **Convert type of Price to float**

# In[8]:


#Renaming the column name Price
vehicle_data.rename({"Price":"Price (Rs)"}, axis = 1, inplace = True)


# In[12]:


#Removing the word Negotiable to cast the string to float
vehicle_data.drop(vehicle_data[vehicle_data['Price (Rs)'] == 'Negotiable'].index, inplace = True)


# In[13]:


#removing spaces,commas,Rs to convert string to float
vehicle_data['Price (Rs)'] = vehicle_data['Price (Rs)'].str.replace('Rs', '').str.replace(',', '').str.replace('.', '').str.strip();


# In[14]:


vehicle_data['Price (Rs)'] = vehicle_data['Price (Rs)'].astype(float)


# **Convert type of Capacity to Int**

# In[15]:


#Renaming
vehicle_data.rename({"Capacity" : "Capacity(cc)"}, axis = 1, inplace = True)


# In[16]:


#Dropping values that are not applicable
vehicle_data.drop(vehicle_data[vehicle_data['Capacity(cc)'] == vehicle_data['Capacity(cc)'].str.startswith('-')].index, inplace = True)
vehicle_data.drop(vehicle_data[vehicle_data['Capacity(cc)'] == '-'].index, inplace = True)
vehicle_data.drop(vehicle_data[vehicle_data['Capacity(cc)'] == 'Manual'].index, inplace = True)
vehicle_data.drop(vehicle_data[vehicle_data['Capacity(cc)'] == 'Automatic'].index, inplace = True)


# In[17]:


vehicle_data.drop(vehicle_data[vehicle_data['Year'] == 0].index, inplace = True)
vehicle_data.drop(vehicle_data[vehicle_data['Year'] >= 2022].index, inplace = True)


# In[18]:


#Converting Capacity to Integer
vehicle_data['Capacity(cc)'] = vehicle_data['Capacity(cc)'].str.replace('cc', '').str.replace(',', '').str.replace('.', '').str.strip().astype('int64');


# **Convert type of Capacity to Int**

# In[19]:


#Renaming
vehicle_data.rename({"Mileage" : "Mileage(km)"}, axis = 1, inplace = True)


# In[20]:


vehicle_data.drop(vehicle_data[vehicle_data['Mileage(km)'] == '-'].index, inplace = True)


# In[21]:


#Converting Mileage to Integer
vehicle_data['Mileage(km)'] = vehicle_data['Mileage(km)'].str.replace('km', '').str.replace(',', '').astype('int64')


# In[22]:


#Changing the values of column Title
vehicle_data['Title'] = vehicle_data['Brand'] + " "+ vehicle_data['Model']


# In[23]:


#Conveting type of Published_date to datetime
vehicle_data['published_date'] = pd.to_datetime(vehicle_data['published_date'])


# In[24]:


vehicle_data.drop(["Sub_title",'Description'], axis = 1, inplace = True)


# In[25]:


vehicle_data.shape


# In[26]:


vehicle_data.describe()


# In[27]:


plt.figure(figsize=(10,5))
v= vehicle_data.corr()
sns.heatmap(v,cmap="RdBu",annot=True)
v


# ## Data Visualizaton

# In[28]:


## Top 5 sellers

top_10_seller = vehicle_data[["Title", "Seller_name"]].groupby("Seller_name").agg(['count'])['Title']['count'].sort_values(ascending=False)[:10]
fig = sns.barplot( x = top_10_seller.index, y = top_10_seller.values, color = 'blue', palette = 'hls')
fig.set_xticklabels(labels=top_10_seller.index , rotation=45)
fig.set_ylabel("Number of Ads")
fig.set_xlabel("Seller Names")
fig.set_title("Top 10 Sellers with most number of ads");


# In[29]:


#Top 5 cars for sale
top_5_car_brand = vehicle_data[["Title", "Brand"]].groupby("Brand").agg(['count'])['Title']['count'].sort_values(ascending=False)[:5]
fig = sns.barplot( x = top_5_car_brand.index, y = top_5_car_brand.values, color = 'blue', palette = 'Paired')
fig.set_xticklabels(labels=top_5_car_brand.index , rotation=45)
fig.set_ylabel("Number of Cars")
fig.set_xlabel("Car Brand")
fig.set_title("Top 5 Most Car For SAle");


# In[30]:


vehicle_data['Year'] = pd.to_datetime(vehicle_data['Year'].astype(str)).values
df = vehicle_data[["Year", "Price (Rs)"]].groupby('Year').agg({"Price (Rs)":"mean"})


# In[31]:


sns.lineplot(data=df, x="Year", y="Price (Rs)").set(title="Distribution of Average Price of Cars over the Years");


# In[32]:


df_2 = pd.DataFrame(vehicle_data.groupby("Condition")                          ["Title"].count()).reset_index().rename({"Condition":"Condition"},axis=1);


# In[33]:


df_2.drop(df_2[df_2['Condition'] == 'e'].index, inplace = True)
df_3 = df_2.replace('Recondition','Reconditioned')
df_4 = pd.DataFrame(df_3.groupby("Condition")                          ["Title"].sum()).reset_index().rename({"Condition":"Condition"},axis=1);
df_4


# In[34]:


y = df_4["Title"]
lbl = df_4["Condition"]
epld = [0.1,0.8,0.5]
plt.pie(y, labels=lbl, startangle = 180, explode = epld, autopct='%1.2f%%')
plt.title("Sum of sale prices according to vehicle types")
plt.show();


# In[35]:


df_5 = pd.DataFrame(vehicle_data.groupby("Fuel")                          ["Title"].count()).reset_index().rename({"Condition":"Condition"},axis=1);
df_6 = df_5.loc[df_5['Fuel'].isin(['Diesel','Hybrid','Petrol'])]
df_6


# In[36]:


x = df_6["Fuel"]
y = df_6["Title"]

plt.bar(x,y)
plt.title("Distribution of cars over Fuel type")
plt.xlabel("Fuel type")
plt.ylabel("Number of vehicles")
plt.xticks(fontsize=14)
plt.show();


# In[ ]:





# ## Model Building

# ### Linear Regression

# In[37]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[38]:


df0 = vehicle_data


# In[39]:


df = df0.sample(frac=0.1)


# In[40]:


def outlier_detection(colname):
  upper_limit = df[colname].mean() + 3*df[colname].std()
  lower_limit = df[colname].mean() - 3*df[colname].std()
  return upper_limit, lower_limit


# In[41]:


df.dtypes


# In[42]:


cleaned_df = df[(df["Price (Rs)"] > outlier_detection("Price (Rs)")[1]) & (df["Price (Rs)"] < outlier_detection("Price (Rs)")[0])]
cleaned_df = df[(df["Capacity(cc)"] > outlier_detection("Capacity(cc)")[1]) & (df["Capacity(cc)"] < outlier_detection("Capacity(cc)")[0])]
cleaned_df = df[(df["Mileage(km)"] > outlier_detection("Mileage(km)")[1]) & (df["Mileage(km)"] < outlier_detection("Mileage(km)")[0])]


# In[43]:


COLS_NEEDED = ["Title", "Condition", "Seller_name", "Price (Rs)", "Capacity(cc)", "Mileage(km)",
               "Transmission", "Seller_type"]
df1 = cleaned_df[COLS_NEEDED]


# In[44]:


COLS_TO_ONE_HOT = ["Title", "Condition", "Seller_name", 'Transmission']
df2 = df1[COLS_TO_ONE_HOT]


# In[46]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore")
ohe_values = ohe.fit_transform(df2).toarray()
ohe_columns = ohe.get_feature_names_out()
df4 = pd.DataFrame(ohe_values, columns=ohe_columns)


# In[48]:


df4.head()


# In[49]:


df1.head()


# In[50]:


df1[["Capacity(cc)","Mileage(km)"]]


# In[51]:


from sklearn.preprocessing import Normalizer


# In[52]:


norm = Normalizer()


# In[53]:


norm_values = norm.fit_transform(df[["Capacity(cc)", "Mileage(km)"]])
norm_cols = ["norm_Capacity(cc)", "norm_Mileage(km)"]


# In[54]:


df5 = pd.DataFrame(norm_values, columns=norm_cols)


# In[55]:


df5.head()


# In[65]:


final_df = pd.concat([df4, df5], axis=1)


# In[61]:


X = final_df
y = df1["Price (Rs)"].fillna(value=np.mean(df1["Price (Rs)"]))


# In[66]:


X.drop(X.tail(10).index,
        inplace = True)


# In[67]:


X.shape


# In[68]:


y.shape


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred_values = lin_reg.predict(X_test)


# In[70]:


final_df.columns


# In[71]:


mean_squared_error(y_test, pred_values)


# In[72]:


final_df.shape


# In[73]:


def final_model(model):
  model = model()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  error_term = mean_squared_error(y_test, pred_values)
  print("Error: ", error_term)


# In[74]:


final_model(LinearRegression)


# In[ ]:




