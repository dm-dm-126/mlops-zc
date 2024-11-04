#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd
import sklearn

sklearn.__version__


# In[5]:


year = 2023
month = 3

input_file = f'data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[7]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[9]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[11]:


df = read_data(input_file)


# In[15]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
y_pred.std() # Q1. What's the standard deviation of the predicted duration for this dataset?


# In[16]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df.head()


# In[17]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# In[21]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[23]:


# Q2. What's the size of the output file?

get_ipython().system('ls -lh output')


# In[ ]:


# Q3. Now let's turn the notebook into a script. Which command you need to execute for that?
# 

