#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


columns1 = ['ep','time', 'elapsed','x-axis','y-axis', 'z-axis']
columns2 = ['ep1','time1', 'elapsed1','x-axis_g','y-axis_g', 'z-axis_g']


# In[3]:


df1 = pd.read_csv('datasets/Sumit_task2_batch1_Accelerometer.csv', header = None, names = columns1)
df2 = pd.read_csv('datasets/Sumit_task2_batch1_Gyroscope.csv', header = None, names = columns2)
df3 = pd.read_csv('datasets/Sumit_task2_batch2_Accelerometer.csv', header = None, names = columns1)
df4 = pd.read_csv('datasets/Sumit_task2_batch2_Gyroscope.csv', header = None, names = columns2)
df5 = pd.read_csv('datasets/Sumit_walking_Accelerometer.csv', header = None, names = columns1)
df6 = pd.read_csv('datasets/Sumit_walking_Gyroscope.csv', header = None, names = columns2)
df7 = pd.read_csv('datasets/Sumit_sitting_Accelerometer.csv', header = None, names = columns1)
df8 = pd.read_csv('datasets/Sumit_sitting_Gyroscope.csv', header = None, names = columns2)


# In[7]:


df_task = pd.concat([df1,df2], axis =1)
del df_task['ep']
del df_task['time']
del df_task['elapsed']
del df_task['ep1']
del df_task['time1']
del df_task['elapsed1']
df_task = df_task.iloc[1:]
df_task['activity'] = 1
df_task.head()


# In[5]:


df_task_1 = pd.concat([df3,df4], axis =1)
del df_task_1['ep']
del df_task_1['time']
del df_task_1['elapsed']
del df_task_1['ep1']
del df_task_1['time1']
del df_task_1['elapsed1']
df_task_1 = df_task_1.iloc[1:]
df_task_1['activity'] = 1
df_task_1.head()


# In[6]:


df_walking = pd.concat([df5,df6], axis =1)
del df_walking['ep']
del df_walking['time']
del df_walking['elapsed']
del df_walking['ep1']
del df_walking['time1']
del df_walking['elapsed1']
df_walking = df_walking.iloc[1:]
df_walking['activity'] = 2
df_walking.head()


# In[7]:


df_sitting = pd.concat([df7,df8], axis =1)
del df_sitting['ep']
del df_sitting['time']
del df_sitting['elapsed']
del df_sitting['ep1']
del df_sitting['time1']
del df_sitting['elapsed1']
df_sitting = df_sitting.iloc[1:]
df_sitting['activity'] = 3
df_sitting.head()


# In[8]:


df_sitting.to_csv(r'Sumit_sitting.csv', header=False, index = False)
df_task.to_csv(r'Sumit_task2.csv', header=False, index = False)
df_task_1.to_csv(r'Sumit_task2_1.csv', header=False, index = False)
df_walking.to_csv(r'Sumit_walking.csv', header=False, index = False)


# In[ ]:




