#!/usr/bin/env python
# coding: utf-8

# James Weaver Project 2 knn classifer

# In[ ]:





# In[1]:


############################################
########## My common Libraries #############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import statsmodels.api as sm
import sklearn
import random as r
###########################################
###########################################

############################################
##########  ignore warnings ################
import warnings
warnings.filterwarnings('ignore')
###########################################
###########################################

random_number = 65421

color_box = ['b','g','r','c','m','y','pink','purple','gold','maroon','lime']


# In[ ]:





# # 1 Collect Data

# In[2]:


data = pd.read_csv('nba_stats_2003_to_20200.csv') # Upload Data
#data.drop(['Unnamed: 29','Unnamed: 30'],axis=1,inplace=True) # Delete the last 2 cols because they are blank
df = data.copy()
df.head()


# In[ ]:





# ### columns

# In[3]:


df.columns


# # 2 Variable Analysis

# In[182]:


df.info


# ### The type of variables in the dataset

# In[6]:


df.dtypes


# ### The data contain floats, integers, and strings(object)

# In[7]:


df.describe()


# ### Inspection the variable values

# In[8]:


for var in df.columns:
    print(var, df[var].unique()[0:20], '\n') 


# It is [0:20] because it will display the first 20 unique variables to see what data we are dealing with.

# ### List of the type or variables

# In[4]:


###### numerical: discrete vs continuous ##### Note that target variable is Playoffs Games Won df[var].dtype=='int64'
target_name = 'Playoffs Games Won'

###### numerical: discrete ###### 
# discrete = [var for var in df.columns if df[var].dtype!='O' and var!=target_name and df[var].nunique()<=10]
discrete1 = [var for var in df.columns if df[var].dtype!='O' and var!=target_name and df[var].dtype=='int64']
discrete2 = [var for var in df.columns if df[var].dtype!='O' and var!=target_name and df[var].nunique()<=10]
discrete = discrete1 + discrete2
discrete = list( dict.fromkeys(discrete))
'''
In this example we we will catorgize the variables as discrete if any of the variables is not a string
and not the target variable 
and contain less than 10 unique values
'''
############################################################################################################################

###### numerical: continuous ######
continuous = [var for var in df.columns if df[var].dtype!='O' and var!=target_name and var not in discrete]
'''
In this example we will catorgize the variables as continuous if any of the variables is not a string
and varible is not the target variable
and variable is not the discrtete varible
'''
#############################################################################################################################

###### mixed varibles ######

mixed = ['Season','POSS']
'''
from the previous tabs, it appeears that Season and POSS are mixed variables because they contain numbers, letters, and symbols.
'''
#############################################################################################################################

# categorical
categorical = [var for var in df.columns if df[var].dtype=='O' and var not in mixed]
'''
any string and not a mixed varibe will be label as categorical
'''
##############################################################################################################################

print(f'There are {len(discrete)} discrete variables')
print(f'There are {len(continuous)} continuous variables')
print(f'There are {len(categorical)} categorical variables')
print(f'There are {len(mixed)} mixed variables')
print(f'\nDiscrete variables:{discrete}\n\nContinuous variables:{continuous}\n\nCategorical variables:{categorical}\n\nMixed variables:{mixed}')


# ### The cardinality of the categorical mixed variables

# In[11]:


'''
the cardinality for Team and season is too high and needs investigation.
'''
df[categorical+mixed].nunique()


# ### Graphs of the continous variables

# In[49]:


df[continuous].describe()


# In[30]:


for i in continuous:
    
    fig, axes = plt.subplots(figsize = (20,7),dpi=200,nrows=1,ncols=3)
    color = r.choice(color_box)

    sns.scatterplot(data=data,x=i,y=target_name,ax=axes[0],facecolor=color);
    axes[0].set_title('scatter plot',color='black', fontsize = 15)

    sns.boxplot(data=data,x=i,ax=axes[1],color=color);
    axes[1].set_title('boxplot',color='black', fontsize = 15)

    sns.histplot(data=data,x=i,bins=15,ax=axes[2],color=color);
    axes[2].set_title('histogram',color='black', fontsize = 15)


# ### Graphs of the discrete variables

# In[51]:


df[discrete].describe()


# In[50]:


for i in discrete:
    
    fig, axes = plt.subplots(figsize = (20,7),dpi=200,nrows=1,ncols=3)
    color = r.choice(color_box)

    sns.scatterplot(data=data,x=i,y=target_name,ax=axes[0],facecolor=color);
    axes[0].set_title('scatter plot',color='black', fontsize = 15)

    sns.boxplot(data=data,x=i,ax=axes[1],color=color);
    axes[1].set_title('boxplot',color='black', fontsize = 15)

    sns.histplot(data=data,x=i,bins=15,ax=axes[2],color=color);
    axes[2].set_title('histogram',color='black', fontsize = 15)


# In[27]:


# dic for outlier style 
flierprops_ = dict(marker = 'o',
                   markerfacecolor='pink',
                   markersize=3,
                   linestyle='none',
                   markeredgecolor='black') 


fig, axes = plt.subplots(figsize = (10,6),dpi=200)

### set graph settings  ###
axes.set_title('Boxplot of the continous variables', # Title of hist
               fontsize = 20, # Font size of title
               color ='black') # Color of title

# boxplot setings 
sns.boxplot(data=df[continuous], # import data
            linewidth=2, # line width
            width=.7,
            flierprops=flierprops_) # outliers style

axes.set_xticklabels(labels = continuous, # name of each tick marks
                             rotation=90, # rotation of each tick mark
                             fontsize = 10, #font size of each tick mark
                             ha= 'center', # hor alig to the right
                             color = 'black');  # color of x-tick labels


# # 3 Feature Engineering

# ## 3.1 Null Analysis

# ### How many observerations are missing for each feature?

# In[821]:


df.isnull().sum()


# Conference Standings, MVP, and Playoffs Games Won contain missing values.

# ### How many observerations are missing for each feature? (as a percentage)

# In[38]:


df.isnull().mean()


# In[43]:


pd.DataFrame(df.isnull().mean()*100)


# ### The rows of missing observations for the Conference Standings Column

# In[824]:


df[df['Conference Standings'].isnull()].head()


# The Conference Standings column have missing values because these observations(teams) did not make it to the playoffs.
# 
# 46.47% of the Conference Standings column contain missing values.

# ### The rows of missing observations for the MVP Column

# In[825]:


df[df['MVP'].isnull()].head()


# The MVP column have missing values because these observations(teams) did not have the MVP for their team.
# 
# 96.65% of the MVP column contain missing values.

# ### The rows of missing observations for the Playoffs Games Won Column

# In[261]:


df[df['Playoffs Games Won'].isnull()].head()


# The Playoffs Games Won column have missing values because these observations(teams) did not make it to the playoffs.
# 
# 46.47% of the Playoffs Games Won column contain missing values.

# ### Target Variable graphs

# In[262]:


sns.boxplot('Playoffs Games Won',data=df);


# In[263]:


sns.histplot(df['Playoffs Games Won'],bins=15);


# In[48]:


fig, axes = plt.subplots(figsize = (20,7),dpi=200)

df['Playoffs Games Won'].value_counts().plot.bar();

axes.set_title('Bar Plot',color='black', fontsize = 25);
axes.set_xlabel('Playoffs Games Won',color='black', fontsize = 15);


# In[ ]:


df['Playoffs Games Won'].value_counts().plot.bar();


# In[5]:


df1 = data.copy()
df1.tail(10)


# ### Puting the target variable into groups

# In[6]:


df1 = data.copy()
step = 0
for i in df1['Playoffs Games Won']:
    if isinstance(i,float):
        if i < 4:
            df1['Playoffs Games Won'][step] = '1st Round'
            step = step + 1
        elif i < 8:
            df1['Playoffs Games Won'][step] = 'Conference Semifinals'
            step = step + 1
        elif i < 12:
            df1['Playoffs Games Won'][step] = 'Conference Finals'
            step = step + 1
        elif i < 16:
            df1['Playoffs Games Won'][step] = 'NBA Finals'
            step = step + 1
        elif i == 16:
            df1['Playoffs Games Won'][step] = 'NBA Champion'
            step = step + 1
        else:
            df1['Playoffs Games Won'][step] = 'No Playoffs'
            step = step + 1
        
        
df1.tail(10)        


# ## Converting the catogorical and mixed features into numbers

# In[7]:


df1[categorical+mixed].head()


# In[8]:


# Lets check cardality 
df1[categorical+mixed].nunique()


# In[56]:


fig, axes = plt.subplots(figsize = (10,6),dpi=200)

df1['TEAM '].value_counts().plot.bar()


# In[868]:


fig, axes = plt.subplots(figsize = (10,6),dpi=200)

df1['Season'].value_counts().plot.bar()


# In[876]:


df1['POSS'].head()


# ### Delete the commas in the POSS column

# In[9]:


# convert the col with the string numbers to float 
df1['POSS'] = df['POSS'].str.replace(',', '').astype(float)


# In[10]:


df1['POSS'].head()


# ### Team and Season are irreveant features.Will not use them for the model 

# In[11]:


df1.isnull().sum()


# ### Ordinal Number Encoding for the target variable

# The target variable has 2 labels. Either the team won the nba champion or not.

# In[12]:


df1 = df1.copy()
step = 0
for i in df1['Playoffs Games Won']:
    if isinstance(i,float):
        if i < 4:
            df1['Playoffs Games Won'][step] = '1st Round'
            step = step + 1
        elif i < 8:
            df1['Playoffs Games Won'][step] = 'Conference Semifinals'
            step = step + 1
        elif i < 12:
            df1['Playoffs Games Won'][step] = 'Conference Finals'
            step = step + 1
        elif i < 16:
            df1['Playoffs Games Won'][step] = 'NBA Finals'
            step = step + 1
        elif i == 16:
            df1['Playoffs Games Won'][step] = 'NBA Champion'
            step = step + 1
        else:
            df1['Playoffs Games Won'][step] = 'No Playoffs'
            step = step + 1



playoffs_rank={'No Playoffs':10,
            '1st Round':0,
            'Conference Semifinals':0,
            'Conference Finals':0,
            'NBA Finals':1,
            'NBA Champion':1}

# will map the games won in playoffs with the playoffs rank number
df1['Playoffs Games Won']=df1['Playoffs Games Won'].map(playoffs_rank)
df1.head()


# ### We will delete all observations that did not make the playoffs

# In[13]:


df1 = df1[df1['Playoffs Games Won'] < 10]


# In[14]:


len(df1)


# Observations are still greater than 230

# In[15]:


df1['Playoffs Games Won'].value_counts()


# In[195]:


df1['Playoffs Games Won'].value_counts().plot.bar();


# The data is imbalance. We will fix this in the training set

# ### Will replace all missing values in the mvp column with zeros

# In[16]:


# fixing MVP col
def replace_nan(df,variable,arbitrary):
    '''
    will replace the na by an arbitrary choice.
    df=dataframe
    variable=the varibale as a string we want
    arbitrary = the arbitrary numeric number 
    '''
    df[variable]=df[variable].fillna(arbitrary)
    
replace_nan(df1,'MVP',0)  


# In[17]:


df1['MVP'].isnull().sum()


# In[18]:


df1.isnull().sum()


# There are no more missing values

# In[65]:


df1.dtypes


# All features are converted into numbers

# In[19]:


rt = 65421


# In[20]:


y_train_os.value_counts().plot.bar()


# In[23]:


X = df1[['Conference Standings', 'WIN%', 'L', 'BLK', 'AST', 'DEFRTG', 'OFFRTG']]
y = df1['Playoffs Games Won']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65421,stratify=y)

# balance dataset
#################################################################################################################
from imblearn.over_sampling import BorderlineSMOTE

number1 = y_train.value_counts()[0] # 
number2 = -int(-(number1*.40)//1) # 

oversample = BorderlineSMOTE(sampling_strategy={0:number1,1:number2},random_state=65421)
X_train_os, y_train_os = oversample.fit_resample(X_train, y_train) # oversample training set
#################################################################################################################

# scale dataset
#################################################################################################################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_X_train_os = scaler.fit_transform(X_train_os) 
scaled_X_test = scaler.transform(X_test)
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################


#################################################################################################################
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn_model.fit(scaled_X_train_os,y_train_os)
#################################################################################################################

# evalute dataset
#################################################################################################################
y_pred = knn_model.predict(scaled_X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score

print(confusion_matrix(y_test,y_pred)) # check the model's confusion matrix
print(classification_report(y_test,y_pred)) # model classication report
print(f1_score(y_test,y_pred)) # f1 score
#################################################################################################################


# In[173]:


listt = list(X_train.columns)
df_scaled_X_train_os = pd.DataFrame(scaled_X_train_os,columns=listt)
df_y_train_os = pd.DataFrame(y_train_os,columns=['Playoffs Games Won'])
new_df = df_scaled_X_train_os
new_df['Target'] = df_y_train_os

fig, axes = plt.subplots(figsize = (10,6),dpi=200)
corr_Matrix = new_df.corr()
sns.heatmap(corr_Matrix, annot=True)


# In[180]:


abs(corr_Matrix['Target']).sort_values(ascending=False)


# In[ ]:





# In[ ]:





# In[24]:


#################################################################################################################
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn_model.fit(scaled_X_train_os,y_train_os)
#################################################################################################################

# evalute dataset
#################################################################################################################
y_pred = knn_model.predict(scaled_X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(confusion_matrix(y_test,y_pred)) # check the model's confusion matrix
print(classification_report(y_test,y_pred)) # model classication report
#################################################################################################################




# Stratified K-Fold Cross-Validation

from sklearn.model_selection import StratifiedKFold,cross_val_score

skfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=65421)
# skfold=StratifiedKFold(n_splits=5)
# shuffle=True...will shuffle data before valdiation

scores=cross_val_score(knn_model,scaled_X_train_os,y_train_os,cv=skfold,scoring='accuracy')
print(scores)
print(np.mean(scores))
print(np.std(scores))


# In[92]:


from sklearn.model_selection import TimeSeriesSplit


# In[93]:


ts_fold = TimeSeriesSplit(5)


# In[89]:


scores=cross_val_score(knn_model,scaled_X_train_os,y_train_os,cv=ts_fold,scoring='accuracy')


# In[44]:


scores


# In[101]:


from statsmodels.tsa.arima_model import ARIMA

ts_fold = TimeSeriesSplit(5).split(df1['OREB%'])


# In[103]:


model = ARIMA(df1['OREB%'],order =(1,1,1) ).fit()


# In[104]:


model.summary()


# In[107]:


scores=cross_val_score(model,X=df1['OREB%'],cv=TimeSeriesSplit(5).split(y),scoring='accuracy')


# In[68]:





# In[82]:


ts_fold


# In[ ]:




