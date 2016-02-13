#---------
# BNP ParisBas Kaggle competition 
#---------

import pandas as pd
import numpy as np 
import matplotlib as mplt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix


#import train & test datafiles
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

#decription of data
df_train.head(5)
df_train.describe()
#target group split
df_train["target"].value_counts()

#get all variable names in a list
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_names=df_train.select_dtypes(numerics)
num_names=list(num_names.columns.values)
#get all categorical names in list
catdf = df_train.select_dtypes(exclude=numerics)
cat_names=list(catdf.columns.values)

print(num_names)
print(cat_names)


#basic recoding of variable - numerics at mean and categoricals at 'UNKNOWN'
for var in num_names:
  mean=df_train[var].mean()
  #print(mean)
  df_train[var].fillna(mean,inplace=True)

#filling missing data (for a list of categorical variables)
for var in cat_names:
  df_train[var].fillna('unknown',inplace=True)


#categorical variables must be encoded as sklearn uses array
le = preprocessing.LabelEncoder()
#convert into numbers
for var in cat_names:
  df_train[var] = le.fit_transform(df_train[var])

# code to convert back (deactivated here as I don't need this)
#for var in cat_names:
# df_train[var] = le.inverse_transform(df_train[var])


#drop ID from df 
df_train.drop('ID', axis=1, inplace=True)

#sampling
df_train['is_train'] = np.random.uniform(0, 1, len(df_train)) <= .75
df_train_train = df_train[df_train['is_train']==True]
df_train_test = df_train[df_train['is_train']==False]
df_train["is_train"].value_counts()

#create features list
features = df_train_train.columns[1:132]
features


#simple Random Forest 
clf = RandomForestClassifier(n_jobs=-1,n_estimators = 150,min_samples_split=2)
clf.fit(df_train_train[features], df_train_train['target'])

#apply clsf on test sample
preds = clf.predict(df_train_test[features])

#confusion matrix
pd.crosstab(df_train_test['target'], preds, rownames=['actual'], colnames=['preds'])

#if happy with results apply preds on official test file for submission

#prepare the test dataframe for scoring
num_names.remove('target');
for var in num_names:
  mean=df_test[var].mean()
  #print(mean)
  df_test[var].fillna(mean,inplace=True)

for var in cat_names:
  df_test[var].fillna('unknown',inplace=True)
  df_test[var] = le.fit_transform(df_test[var])


#predictions
preds_forsub=clf.predict(df_test[features])