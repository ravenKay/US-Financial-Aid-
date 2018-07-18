#PART A - LOAD IN PACKAGES, IMPORT DATASET
import numpy as np 
import pandas as pd
import sklearn
import csv
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, svm, metrics, neighbors
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV


#Load data
names = ['country_id',	'region_id',	'income_group_id',	'implementing_agency_id',	'implementing_subagency_id',	'channel_category_id',	'channel_subcategory_id',	'channel_id',	'dac_category_id',	'dac_sector_code',	'dac_purpose_code',	'funding_agency_id',	'assistance_category_id',	'aid_type_group_id',	'activity_id',	'activity_start_date',	'activity_end_date',	'transaction_type_id',	'fiscal_year',	'current_amount',	'constant_amount',	'USG_sector_id',	'submission_id']
aid_data = pd.read_csv(r'us_foreign_aid_short.csv', names = names)
missing_count = aid_data.isnull().sum()
print(missing_count)
df = aid_data.convert_objects(convert_numeric =True)
print(df.dtypes)

#Move prediction to the end
cols = df.columns.tolist()
if "region_id" in cols: 
	names.append('region_id')


#PART B - CHECK FEATURES, DIMENSIONS, MISSING VALUES, ETC. 
#Exploratory Data Analysis 
print(aid_data.shape)
print(aid_data.head(5))
print(aid_data.describe())
print(aid_data.columns)
print(len(aid_data))

#Check list of countries and columns for data quality concerns 
unique = aid_data.country_id.unique()

#drop the start and end dates 
df = df.drop(['activity_end_date', 'activity_start_date'], axis =1)
print(df.dtypes)


#PART C - EXPLORATORY DATA ANALYSIS 
print("Pearson and Spearson correlations are (respectively): ")
print(df.corr())
print(df.corr('spearman'))


#PART D - MODELING - Select, test, and evaluate 
df = pd.read_csv('us_foreign_aid_short.csv')
df = df.drop(['activity_end_date', 'activity_start_date'], axis =1)
df = df.fillna(-99999)

#K-Nearest Neighbors Module
X= np.array(df.drop(['region_id'],1))
y= np.array(df['region_id'])
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.8)
clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train,y_train)
knn_accuracy = clf.score(X_test,y_test)
#KNN evaluate: 
print('The accuracy of the KNN test was :') 
print(knn_accuracy)
y_prediction = clf.predict(X_test)
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))


#SVM Model 
X= np.array(df.drop(['region_id'],1))
y= np.array(df['region_id'])
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = svm.SVC()
clf.fit(X_train,y_train)
svm_accuracy = clf.score(X_test,y_test)
#SVM evaluate: 
print('The accuracy of the SVM test was :') 
print(svm_accuracy)
y_prediction = clf.predict(X_test)
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))


#Random Forest 
#Use Recursive Feature Elimination 
X= np.array(df.drop(['region_id'],1))
y= np.array(df['region_id'])
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
m = RFECV(RandomForestClassifier(n_jobs=-1), scoring='accuracy',verbose=1)
m.fit(X, y)
k = m.score(X, y)
print("Using recursive feature elimination of a random forest, best model produces the folloiwng accuracy: ")
print(k)

