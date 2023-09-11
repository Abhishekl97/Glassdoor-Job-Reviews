#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# Plot settings
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 10


# Reading the dataset which is a **.csv file** and creating a dataframe. 

# In[74]:


data = pd.read_csv('glassdoor_reviews.csv')
data.head()


# ## Section 1: Data Cleaning, Data Pre-processing and Data Analysis
# 
# The dataset contains columns **recommend, ceo_approve, outlook** with column values **v, r, x, o**. We will convert these values in terms of the true sentiment that they represent. Below are the conversions:
# 
# * v - Positive 
# * r - Neutral
# * x - Negative
# * o - No opinion

# In[75]:


data.replace({'recommend' : {'v':'Positive','r':'Neutral','x':'Negative','o':'No Opinion'},
            'ceo_approv' : {'v':'Positive','r':'Neutral','x':'Negative','o':'No Opinion'},
            'outlook' : {'v':'Positive','r':'Neutral','x':'Negative','o':'No Opinion'}}, inplace=True)
data.head()


# The column **current** represents the current status of the employee. To get some context we will change its name to **Employee Status** 

# In[76]:


data.rename(columns={'current':'Employee Status'}, inplace=True)
data.head()


# As we will be performing our analysis on job reviews in the UK(United Kingdom). Let's extract the number of countries from the **location** column to understand the number of values for reviews.

# In[77]:


# Extracting the country names from location using regex expression.
# Assigning them to a new column 'country' in the dataframe. 

data['location']=data['location'].fillna('')
data['city'] = data['location'].str.extract(r'(\w+\s*\w*\s*\w*),*').fillna('')
data['country'] = data['location'].str.extract(r'(\w+\s*\w*)$').fillna('')
data.head()
#data['country'].value_counts().head(10)


# The **date_review** column contains the date in **'yyyy-mm-dd'** format. We will extract the year and month from that column to understand the range of years that we have reviews from and to further use this information for data analysis.

# In[78]:


# Extracting values from the date_review column using regex and assigning it to new columns
# 'year' and 'month' in the dataset respectively 

data['year']= data['date_review'].str.extract(r'(\d\d\d\d)-').fillna('')
data['month']= data['date_review'].str.extract(r'\d\d\d\d-(\d\d)-').fillna('')
data.head()


# As we see below we have reviews from **2008** to **2021** 

# In[79]:


data['year'].unique()


# We will **update** the dataset such that it contains reviews only from UK(United Kingdom).

# In[80]:


# Forming a dataset with reviews only from UK(United Kingdom)

data = data.query('country == "Scotland" or country == "England"')
data.head()


# We will be replacing all the Non-numeric values in the features with 0 and then modifying the dataframe and we will filter out the reviews having 0 values in all the features.

# In[81]:


# Replacing the values using fillna() and modifying the data using a query

data['culture_values'] = data['culture_values'].fillna(0)
data['career_opp'] = data['career_opp'].fillna(0)
data['work_life_balance'] = data['work_life_balance'].fillna(0)
data['diversity_inclusion'] = data['diversity_inclusion'].fillna(0)
data['comp_benefits'] = data['comp_benefits'].fillna(0)
data['senior_mgmt'] = data['senior_mgmt'].fillna(0)

data=data.query('culture_values != 0.0 & career_opp != 0.0 & work_life_balance != 0.0 & comp_benefits != 0.0 & senior_mgmt != 0.0')
data.head()


# To efficiently analyse our data we will create a function which will take a **dataset, column, and count variable** and return a dataframe which has values in the column more than the mentioned threshold (count).
# 
# Below, we will process our data such that it has cities with 200 or more records. This will help us to eliminate cities which do not have significant amount of records.

# In[82]:


#Defining a funtion which returns records having values in the columns more than the mentioned threshold

def top_values(df,col,count):
    df_data= df[col].value_counts().to_frame()
    df_data['new_col']=df_data.index
    return df[df[col].isin(df_data[df_data[col]>=count]['new_col'])]

#Converting all values of Firm to title case
data['firm']=data['firm'].str.title()

#Getting cities with more than 200 records

data=top_values(data,'city',200)
data.head()


# Now, we will process our data to have firms with 50 or more reviews in order to have considerable amount of reviews for each firm.

# In[83]:


#Getting records for firms with more than 50 ratings
data=top_values(data,'firm',50)

# Getting the counts of rating for each firm
data_counts = data[['firm','overall_rating']].groupby(['firm','overall_rating']).agg({'overall_rating':'count'}).rename(columns={'overall_rating':'count'})
data_counts


# To understand the number of ratings received for each firm, we will plot a pivot table and analyse ratings received for each firm and the **Total** ratings for the firm.  

# In[84]:


pivot_table=pd.pivot_table(data,
               index='firm',
               columns='overall_rating',
               values='year',
               aggfunc='count',
               margins=True).rename(columns={'All':'Total'}).fillna(0).astype('int')
#pivot_table=pivot_table.rename({pivot_table.index[-1]:'Total'})
pivot_table=pivot_table.drop('All')
pivot_table


# To analyse the **top 10 best and top 10 worst** rated firms in the dataset we will use our pivot table.

# In[85]:


pivot_table['5_star_prop']=pivot_table[5]/pivot_table['Total']

top_10_best=pivot_table['5_star_prop'].sort_values(ascending=False).head(10).index.to_numpy()

print('Top 10 firms with most 5 star ratings:')

print('\n',top_10_best)

pivot_table['average']=(pivot_table[1]+pivot_table[2]*2+pivot_table[3]*3+pivot_table[4]*4+pivot_table[5]*5)/(pivot_table['Total'])
print('\nTop 10 firms with best ratings:')
pivot_table['average'].sort_values(ascending=False).head(10).to_frame()


# In[86]:


pivot_table['1_star_prop']=pivot_table[1]/pivot_table['Total']
top_10_worst=pivot_table['1_star_prop'].sort_values(ascending=False).head(10).index.to_numpy()
print('Top 10 firms with most 1 star ratings:')
print('\n',top_10_worst)

print('\nTop 10 firms with worst ratings:')
pivot_table['average'].sort_values().head(10).to_frame()


# To visualize the trends in the relation between the Overall Rating and Features for prediction we plot a lineplot.

# In[87]:


sns.lineplot(data=data,x='overall_rating',y='culture_values',label='Culture Values');
sns.lineplot(data=data,x='overall_rating',y='career_opp', label = 'Career Opportunities');
sns.lineplot(data=data,x='overall_rating',y='work_life_balance', label = 'Work Life Balance');
sns.lineplot(data=data,x='overall_rating',y='comp_benefits', label = 'Compensation Benefits');
sns.lineplot(data=data,x='overall_rating',y='senior_mgmt', label = 'Senior Management');
plt.xlabel('Overall Rating');
plt.ylabel('Features');
plt.title('Overall Rating vs Features for Prediction');
plt.legend();


# ## Section 2: Training a model and predicting using Linear Regression method
# 
# 
# 
# We will be predicting our model on the **overall rating** column with feature columns **work_life_balance, culture_values, career_opp, comp_benefits and senior_mgmt**.

# Splitting the data into **train** and **test** sets with 80% data assigned to the train set and 20% assigned to the test set.

# In[88]:


def train_test_split(data):
    data_len = data.shape[0]
    shuffled_indices = np.random.permutation(data_len)
    train_indices = shuffled_indices[0:int(data_len*0.8)]
    test_indices = shuffled_indices[int(data_len*0.8):]
    train = data.iloc[train_indices,:]
    test = data.iloc[test_indices,:]
    return train,test
train, test = train_test_split(data)
print('Count of train dataset: ', len(train))
print('Count of test dataset : ', len(test))


# Now, we use the train data to build the model using **Linear Regression method** and analyse the overall rating of the firm and its prediction.

# In[89]:


from sklearn import linear_model as lm

model = lm.LinearRegression(fit_intercept=True)

def get_feature_output(data,pred_col):
    features = data.drop(columns=[pred_col]).to_numpy()
    actual_output = data.loc[:, pred_col].to_numpy()
    return features, actual_output

train_feature,train_output=get_feature_output(train.loc[:,['overall_rating','culture_values','career_opp','work_life_balance','comp_benefits','senior_mgmt']],'overall_rating')
train_prediction = model.fit(train_feature,train_output).predict(train_feature)
train_prediction =np.round(train_prediction,decimals=0)
#(train['overall_rating']-train_prediction).to_frame().value_counts()
train_data=train.loc[:,['firm','overall_rating']]
train_data['prediction']=train_prediction
train_data['deviation']=(train['overall_rating']-train_data['prediction']).to_frame()
train_data.head()
#train_data['deviation'].value_counts()


# Below, we calculate the train **RMSE loss**, **r2 score** and **theta** values for the model.

# In[90]:


#Getting training RMSE
train_rmse = np.sqrt(np.mean((train_output - train_prediction)**2))
train_rmse


# In[91]:


# Getting r2 score
np.var(train_prediction)/np.var(train_output)


# In[92]:


train_theta0 = model.intercept_
train_theta1, train_theta2, train_theta3, train_theta4, train_theta5= model.coef_

print("Model Parameters\nθ0: {}\nθ1: {}\nθ2: {}\nθ3: {}\nθ4: {}\nθ5: {}".format(train_theta0,train_theta1,train_theta2, train_theta3, train_theta4, train_theta5))


# Now, we use the test data to analyse the overall rating of the firm and its prediction.

# In[93]:


test_feature,test_output=get_feature_output(test.loc[:,['overall_rating','culture_values','career_opp','work_life_balance','comp_benefits','senior_mgmt']],'overall_rating')
test_prediction = model.predict(test_feature)
test_prediction =np.round(test_prediction,decimals=0)
#(test['overall_rating']-test_prediction).to_frame().value_counts()
test_data=test.loc[:,['firm','overall_rating']]
test_data['prediction']=test_prediction
test_data['deviation']=(test['overall_rating']-test_data['prediction']).to_frame()
test_data.head()
#test_data['deviation'].value_counts()


# Calculating the test **RMSE loss**, **r2 score** and **theta** values for the model.

# In[94]:


test_rmse = np.sqrt(np.mean((test_output - test_prediction)**2))
test_rmse


# In[95]:


# Getting r2 score
np.var(test_prediction)/np.var(test_output)


# In[96]:


test_theta0 = model.intercept_
test_theta1, test_theta2, test_theta3, test_theta4, test_theta5= model.coef_

print("Test Model Paramters\nθ0: {}\nθ1: {}\nθ2: {}\nθ3: {}\nθ4: {}\nθ5: {}".format(test_theta0,test_theta1, test_theta2, test_theta3, test_theta4, test_theta5))


# Finally, we use the complete data to analyse the overall rating of the firm and its prediction.

# In[97]:


data_feature,data_output=get_feature_output(data.loc[:,['overall_rating','culture_values','career_opp','work_life_balance','comp_benefits','senior_mgmt']],'overall_rating')
data_prediction = model.predict(data_feature)
data_prediction =np.round(data_prediction,decimals=0)
(data['overall_rating']-data_prediction).to_frame().value_counts()

new_data=data.loc[:,['firm','overall_rating']]
new_data['prediction']=data_prediction
new_data['deviation']=(new_data['overall_rating']-new_data['prediction']).to_frame()
#new_data['deviation'].value_counts()


# Calculating the **RMSE loss**, **r2 score** and **theta** values for the model.

# In[98]:


data_rmse = np.sqrt(np.mean((data_output - data_prediction)**2))
data_rmse


# In[99]:


np.var(data_prediction)/np.var(data_output)


# In[100]:


theta0 = model.intercept_
theta1, theta2, theta3, theta4, theta5= model.coef_

print("Full Model Paramters\nθ0: {}\nθ1: {}\nθ2: {}\nθ3: {}\nθ4: {}\nθ5: {}".format(theta0,theta1, theta2, theta3, theta4, theta5))


# Here, we are plotting a scatter plot of the actual output to the residual value(actual - prediction)

# In[101]:


plt.scatter(x=test_data['overall_rating'],y=test_data['deviation']);
plt.title('Relation of Actual Output to the Deviation from the Actual Output');
plt.xlabel('Overall Rating');
plt.ylabel('Residual');


# In[102]:


sns.lmplot(data=test_data,x='overall_rating',y='prediction');
plt.title('Relation of Actual Output to the Predicted Output');
plt.xlabel('Overall Rating');
plt.ylabel('Prediction');


# ## Section 3: Sentiment Analysis
# 
# To perform sentiment analysis we will create a new dataframe **updated_data** containing columns **firm, year and headline**. We will be performing our analysis on the **headline** column.
# 
# Before performing the sentiment analysis, first, we need to remove all the punctuations from the headline column in the dataframe.

# In[103]:


# Creating a updated_data dataframe 
# Removing punctuations from the headline column in the dataframe.

updated_data = data[['firm','year','headline']].dropna().reset_index(drop=True)
updated_data['headline'] = updated_data['headline'].str.lower().replace(r'[^\w\s\t\n]','', regex=True)
updated_data.head()


# Now, we perform sentiment analysis by using the SentimentIntensityAnalyzer class from nltk.

# In[104]:


# we need to pip install nltk and vaderSentiment for Sentiment analysis

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# We will create a function that will take a dataframe and a column as an input to perform sentiment analysis and produce an output containing 4 columns with **negative, neutral, positive and compound** score for each row in the headline column.

# In[105]:


sent = SentimentIntensityAnalyzer()

def Sentiment_Analyzer(df,column):
    df['negative'] = column.apply(lambda x:sent.polarity_scores(x)['neg'])
    df['neutral'] = column.apply(lambda x:sent.polarity_scores(x)['neu'])
    df['positive'] = column.apply(lambda x:sent.polarity_scores(x)['pos'])
    df['compound'] = column.apply(lambda x:sent.polarity_scores(x)['compound'])
    return df

Sentiment_Analyzer(updated_data,updated_data["headline"])
updated_data


# # THANK YOU!!!
