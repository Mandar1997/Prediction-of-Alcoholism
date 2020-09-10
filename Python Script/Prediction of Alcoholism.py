# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:14:05 2020

@author: Mandar
"""

#import all libraries
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report

#load a single csv file
onedata=pd.read_csv('F:\Data Science\My\Task\SMNI_CMI_TRAIN\Train\Data1.csv')

##########################         load the train data     #########################
#read the directory in which all train files are stored
stock_file=sorted(glob('F:\Data Science\My\Task\SMNI_CMI_TRAIN\Train\Data*.csv'))

#merge all the csv files into a single dataframe
train_data=pd.concat((pd.read_csv(file).assign(filename=file)for file in stock_file),ignore_index=True)

#find info of the dataset
train_data.info()
train_data_decr=train_data.describe()

#view first 5 rows of the dataset
train_head_data=train_data.head()

#find unique elements of columns
train_data['sensor position'].unique()
train_data['channel'].unique()
train_data['matching condition'].unique()
train_data['subject identifier'].unique()

#drop unwanted columns 
train_data=train_data.drop(['Unnamed: 0','filename','name','sensor position'],axis=1)

#apply label encoder to get numeric values
label_encoder = preprocessing.LabelEncoder()
train_data['matching condition']= label_encoder.fit_transform(train_data['matching condition'])
train_data['subject identifier']= label_encoder.fit_transform(train_data['subject identifier'])

#check for null values
sns.heatmap(train_data.isnull(), cbar=False)
train_data.isnull().values.any()

#drop duplicate data
train_data=train_data.drop_duplicates()

#correlation
sns.heatmap(train_data.corr())
train_data_corr=train_data.corr()
plt.figure(figsize = (12,10))
sns.heatmap(train_data_corr,square=True,annot=True,linewidths=4,linecolor='k')

#############################            plots             ############################
#factorplot
sns.catplot('sensor value','subject identifier',data = train_data, height = 4, aspect = 3)
sns.catplot('matching condition','subject identifier',data = train_data, height = 4, aspect = 3)

#barplot
sns.barplot(x='subject identifier', y='time', data=train_data)

#histogram
plt.figure(figsize=(10,7))
plt.hist(train_data['subject identifier'],color='orange', bins=20)
plt.show()

#distplot
plt.figure(figsize = (12,6))
sns.distplot(train_data['subject identifier'],kde=True,bins=20)
plt.xlabel("subject identifier")
plt.title("Destribution of frequency")
plt.grid(linestyle='-.',linewidth = .5)

#line plot
sns.lmplot(x='sensor value',y='subject identifier',data=train_data,markers='.')

#box plot
plt.figure(figsize = (12,6))
sns.boxplot(train_data['subject identifier'],data=train_data)
plt.xlabel("subject identifier")
#plt.xlim(10,40)
plt.grid(linestyle='-.',linewidth = .5)

#scatter plot
plt.scatter(train_data['matching condition'],train_data['subject identifier'],color="blue")
plt.scatter(train_data['sensor value'],train_data['subject identifier'],color="blue")

############################         load the test data       ##########################
#read the directory in which all test files are stored
stock_file_test=sorted(glob('F:\Data Science\My\Task\SMNI_CMI_TEST\Test\Data*.csv'))

#merge all the csv files into a sinfle dataframe
test_data=pd.concat((pd.read_csv(file).assign(filename=file)for file in stock_file),ignore_index=True)

#test data info
test_data.info()

#drop unwanted columns 
test_data=test_data.drop(['Unnamed: 0','filename','name','sensor position'],axis=1)

#find the unique values
test_data['matching condition'].unique()
test_data['subject identifier'].unique()

#apply label encoder to get numeric values
label_encoder_test = preprocessing.LabelEncoder()
test_data['matching condition']= label_encoder_test.fit_transform(test_data['matching condition'])
test_data['subject identifier']= label_encoder_test.fit_transform(test_data['subject identifier'])

#drop duplicate data
test_data=test_data.drop_duplicates()

###############################       define training and testing sets      ###########################
X_train=train_data.drop("subject identifier",axis=1)
Y_train=train_data["subject identifier"].values.reshape(-1,1)
X_test=test_data.drop("subject identifier",axis=1)
Y_test=test_data["subject identifier"].values.reshape(-1,1)

#We can see that all the data given is in a uniform state. Therefore, we can use Logistic Regression and Decision Tree.
###################################        Logistic Regression         #########################
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

#Y_pred=Y_pred.reshape(-1,1)

#regression score
logreg.score(X_train, Y_train)
logreg.score(X_test, Y_test)
#logreg.score( Y_test,Y_pred)

#export predicted data into a csv file
logreg_sub = pd.DataFrame({"subject identifier": Y_pred})
logreg_sub.to_csv('Prediction of Logistic Regression Model.csv', index=False)

#classification report
cr_logreg=classification_report(Y_test, Y_pred)

###################################   confusion matrix for Logistic Reg     ##########################
cm_logreg=confusion_matrix(Y_test, Y_pred)

#heat map of confusion matrix
plt.figure(figsize = (12,6))
sns.heatmap(cm_logreg,square=True,annot=True,linewidths=4,linecolor='k')
plt.xlabel('actual')
plt.ylabel("predicted")

#export the confusion matrix into a csv file
cm_logreg_df=pd.DataFrame(cm_logreg)
cm_logreg_df.to_csv('Confusion Matrix of Logistic Regression Model.csv', index=False)

##################################            Decision Tree           ############################
dt=DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred=dt.predict(X_test)
#Y_pred=Y_pred.reshape(-1,1)

#regression score
dt.score(X_train, Y_train)
dt.score(X_test, Y_test)

#export predicted data into a csv file
dt_sub = pd.DataFrame({"predicted subject identifier": Y_pred})
dt_sub.to_csv('Prediction of Decision Tree Model.csv', index=False)

#classification report
cr_dt=classification_report(Y_test, Y_pred)

###################################   confusion matrix for Decision Tree     ##########################
cm_dt=confusion_matrix(Y_test, Y_pred)

#heat map of confusion matrix
plt.figure(figsize = (12,6))
sns.heatmap(cm_dt,square=True,annot=True,linewidths=4,linecolor='k')
plt.xlabel('actual')
plt.ylabel("predicted")

#export the confusion matrix into a csv file
cm_dt_df=pd.DataFrame(cm_dt)
cm_dt_df.to_csv('Confusion Matrix of Decision Tree Model.csv', index=False)

###################################       K Fold cross validation     #########################
kf=KFold(n_splits=10)
for train_index,test_index in kf.split(train_data):
    print(train_index,test_index)

def get_score(model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train,Y_train)
    return model.score(X_test,Y_test)
kf_score_logreg=get_score(LogisticRegression(),X_train,X_test,Y_train,Y_test)
kf_score_dt=get_score(DecisionTreeClassifier(),X_train,X_test,Y_train,Y_test)

#################################    stratified K Fold cross validation   #######################
folds=StratifiedKFold(n_splits=10)
skf_score_logreg=cross_val_score(LogisticRegression(), X_train,Y_train)
skf_score_dt=cross_val_score(DecisionTreeClassifier(), X_train,Y_train)
