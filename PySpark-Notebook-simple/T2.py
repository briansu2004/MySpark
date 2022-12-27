#!/usr/bin/env python
# coding: utf-8

# #### In this Video We will Cover
# - PySpark Dataframe 
# - Reading The Dataset
# - Checking the Datatypes of the Column(Schema)
# - Selecting Columns And Indexing
# - Check Describe option similar to Pandas
# - Adding Columns
# - Dropping columns
# - Renaming Columns

# In[3]:


get_ipython().system('pip3 install pyspark pandas')


# In[4]:


import pyspark


# In[5]:


from pyspark.sql import SparkSession


# In[6]:


spark=SparkSession.builder.appName('Dataframe').getOrCreate()


# In[7]:


spark


# In[ ]:


## read the dataset
df_pyspark=spark.read.option('header','true').csv('test1.csv',inferSchema=True)


# In[ ]:


### Check the schema
df_pyspark.printSchema()


# In[ ]:


df_pyspark=spark.read.csv('test1.csv',header=True,inferSchema=True)
df_pyspark.show()


# In[ ]:


### Check the schema
df_pyspark.printSchema()


# In[ ]:


type(df_pyspark)


# In[ ]:


df_pyspark.head(3)


# In[ ]:


df_pyspark.show()


# In[ ]:


df_pyspark.select(['Name','Experience']).show()


# In[ ]:


df_pyspark['Name']


# In[ ]:


df_pyspark.dtypes


# In[ ]:


df_pyspark.describe().show()


# In[ ]:


### Adding Columns in data frame
df_pyspark=df_pyspark.withColumn('Experience After 2 year',df_pyspark['Experience']+2)


# In[ ]:


df_pyspark.show()


# In[ ]:


### Drop the columns
df_pyspark=df_pyspark.drop('Experience After 2 year')


# In[ ]:


df_pyspark.show()


# In[ ]:


### Rename the columns
df_pyspark.withColumnRenamed('Name','New Name').show()


# In[ ]:




