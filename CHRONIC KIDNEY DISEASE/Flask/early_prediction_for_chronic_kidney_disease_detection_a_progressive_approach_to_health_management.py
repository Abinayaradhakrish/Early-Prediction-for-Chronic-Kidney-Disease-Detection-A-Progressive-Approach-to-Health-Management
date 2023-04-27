# -*- coding: utf-8 -*-
"""Early Prediction for Chronic Kidney Disease Detection: A Progressive Approach to Health Management.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1efMQtGHkdDgSQWm5K1gipFBx9BXIBnyV
"""

import pandas as pd #used for data manipulation import numpy as np #used for numerical analysis
from collections import Counter as c# return counts of number of classess import matplotlib.pyplot as plt #used for data Visualization
import seaborn as sns #data visualization Library
import missingno as msno #finding missing values
from sklearn.metrics import accuracy_score, confusion_matrix#model performance
from sklearn.model_selection import train_test_split #splits data in random train and test array from sklearn.preprocessing import LabelEncoder #encoding the levels of categorical features
from sklearn.linear_model import LogisticRegression #Classification ML algorithm
import pickle #Python object hierarchy is converted into a byte stream

data=pd.read_csv('/content/kidney_disease.csv')

data.head()

data.columns

data = data.drop('id', axis=1)

data.columns=['age', 'blood_pressure', 'specific_gravity', 'albumin',
'sugar', 'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood glucose random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetesmellitus', 'coronary artery disease', 'appetite', 'pedal_edema', 'anemia','class'] # manually giving the name of the columns
data.columns

data.info()

data.isnull().any()

data['blood glucose random'].fillna(data['blood glucose random'].mean(), inplace=True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(), inplace=True)
data['blood_urea'].fillna(data['blood_urea'].mean(), inplace=True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(), inplace=True)
#data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(), inplace=True)
data['packed_cell_volume'].apply(lambda x: isinstance(x, (int, float))).unique()
data['packed_cell_volume'] = pd.to_numeric(data['packed_cell_volume'], errors='coerce')
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(), inplace=True)
data['potassium'].fillna(data['potassium'].mean(),inplace=True)
#data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(), inplace=True)
data['red_blood_cell_count'].apply(lambda x: isinstance(x, (int, float))).unique()
data['red_blood_cell_count'] = pd.to_numeric(data['red_blood_cell_count'], errors='coerce')
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(), inplace=True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(), inplace=True)
data['sodium'].fillna(data['sodium'].mean(), inplace=True)
#data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(), inplace=True)
data['white_blood_cell_count'].apply(lambda x: isinstance(x, (int, float))).unique()
data['white_blood_cell_count'] = pd.to_numeric(data['white_blood_cell_count'], errors='coerce')
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(), inplace=True)

data['age'].fillna(data['age'].mode()[0], inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0], inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0], inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0], inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0], inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0], inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0], inplace=True)
data['coronary artery disease'].fillna(data['coronary artery disease'].mode()[0], inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0], inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0], inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0], inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0], inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0], inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0], inplace=True)

catcols=set(data.dtypes[data.dtypes=='0'].index.values)
print(catcols)

for i in catcols:
 print("Continous Columns :",i)
 print(c(data[i]))
 print('*'*120+'\n')

if 'red_blood_cell_count' in catcols:
    catcols.remove('red_blood_cell_count')
if 'packed_cell_volume' in catcols:
    catcols.remove('packed_cell_volume')
if 'white_blood_cell_count' in catcols:
    catcols.remove('white_blood_cell_count')

catcols=['anemia','pedal_edema','appetite','bacteria','class','coronary artery disease','diabetesmellitus','hypertension','pus_cell', 'pus_cell_clumps','red_blood_cells']

from sklearn.preprocessing import LabelEncoder

# Looping through all the categorical columns
for i in catcols:
    print("LABEL ENCODING OF:",i)
    LEi = LabelEncoder() # creating an object of LabelEncoder
    print(c(data[i])) #getting the classes values before transformation
    data[i] = LEi.fit_transform(data[i]) # transforming our text classes to numerical values
    print(c(data[i])) #getting the classes values after transformation
    print("*"*100)

contcols=set(data.dtypes[data.dtypes!='0'].index.values)# #contcols=pd.DataFrame(data, columns=contcols)
print(contcols)

for i in contcols:
 print("Continous Columns :",i)
 print(c(data[i]))
 print('*'*120+'\n')

contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')
print(contcols)

contcols.add('red_blood_cell_count') 
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)

contcols.add('specific_gravity')
contcols.add('albumin')
contcols.add('sugar')
print(catcols)

data['coronary artery disease'] = data['coronary artery disease'].replace('\tno', 'no')
c(data['coronary artery disease'])

data['diabetesmellitus'] = data['diabetesmellitus'].replace(to_replace={"\\tno": "no", "\tyes": "yes", " yes": "yes"})
c(data['diabetesmellitus'])

data.describe()

sns.distplot(data.age)

import matplotlib.pyplot as plt # import the matplotlib libaray
fig=plt.figure(figsize=(5,5)) #plot size
plt.scatter(data['age'],data['blood_pressure'],color='blue')
plt.xlabel('age') #set the label for x-axis
plt.ylabel('blood pressure') #set the label for y-axis
plt.title("age VS blood Scatter Plot") #set a title for the axes

plt.figure(figsize=(20,15),facecolor="white")
plotnumber = 1
for column in contcols: 
  if plotnumber<=11 : 
     ax = plt.subplot(3,4,plotnumber) 
     plt.scatter(data['age'], data[column]) 
     plt.xlabel(column, fontsize=20)
  plotnumber+=1
plt.show()

f,ax=plt.subplots(figsize=(18,10))
sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, linewidths=0.5, linecolor="orange")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

sns.countplot(data['class'])

import pandas as pd

# assume that 'data' is a pandas DataFrame containing your data
print(data.columns)

selcols=['red_blood_cells', 'pus_cell', 'blood glucose random', 'blood_urea', 'pedal_edema', 'anemia', 'diabetesmellitus', 'coronary artery disease']
x=pd.DataFrame(data,columns=selcols)
y=pd.DataFrame(data,columns=['class'])
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classification = Sequential()
classification.add(Dense(30,activation='relu'))
classification.add(Dense(128,activation='relu'))
classification.add(Dense(64,activation='relu'))
classification.add(Dense(32,activation='relu'))
classification.add(Dense(1,activation='sigmoid'))

classification.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

classification.fit(x_train, y_train, batch_size=10, validation_split=0.2, epochs=100)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(x_train,y_train)

y_predict = rfc.predict(x_test)
y_predict_train = rfc.predict(x_train)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=4, splitter='best', criterion='entropy')
dtc.fit(x_train, y_train)

y_predict= dtc.predict(x_test)
y_predict

y_predict_train = dtc.predict(x_train)

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression() 
lgr.fit(x_train,y_train)

from sklearn.metrics import accuracy_score, classification_report
y_predict = lgr.predict(x_test)

y_pred = lgr.predict([[1, 1, 121.000000, 36.0, 0, 0, 1, 0]])
print(y_pred)

y_pred = dtc.predict([[1, 1, 121.000000, 36.0, 0, 0, 1, 0]])
print(y_pred)

y_pred = rfc.predict([[1, 1, 121.000000, 36.0, 0, 0, 1, 0]])
print(y_pred)

classification.save("ckd.h5")

y_pred = classification.predict(x_test)

y_pred

y_pred = (y_pred > 0.5)
y_pred

def predict_exit(sample_value):
    # Convert list to numpy array 
    sample_value = np.array(sample_value)
    
    # Reshape because sample_value contains only 1 record 
    sample_value = sample_value.reshape(1, -1)
    
    # Feature Scaling
    sample_value = sc.transform(sample_value)
    
    return classifier.predict(sample_value)

test=classification.predict([[1,1,121.000000,36.0,0,0,1,0]]) 
if test==1:
    print('Prediction: High chance of CKD!')
else:
    print('Prediction: Low chance of CKD.')

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd

dfs = []
models = [('LogReg', LogisticRegression()), ('RF', RandomForestClassifier()), ('DecisionTree', DecisionTreeClassifier())]

results = []
names = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
target_names = ['NO CKD', 'CKD']

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(model, x_train, y_train, cv=kfold, scoring=scoring)
    clf = model.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name)
    print(classification_report(y_test, y_pred, labels=clf.classes_, target_names=target_names))
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)

final = pd.concat(dfs, ignore_index=True)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_predict)
cm

# Plotting confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for Logistic Regression model') 
plt.show()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for DecisionTreeClassifier')
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels=['no ckd', 'ckd'], yticklabels=['no ckd', 'ckd'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Confusion Matrix for ANN model')
plt.show()

bootstraps = []
for model in list(set(final.model.values)):
    model_df = final.loc[final.model == model]
    bootstrap = model_df.sample(n=30, replace=True)
    bootstraps.append(bootstrap)
bootstrap_df = pd.concat(bootstraps, ignore_index=True)
results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='values')
time_metrics = ['fit_time','score_time'] # fit time metrics
## PERFORMANCE_METRICS
results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get_df without fit data
results_long_nofit = results_long_nofit.sort_values(by='values')
## TIME METRICS
results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
results_long_fit = results_long_fit.sort_values(by='values')

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20, 12))
sns.set(font_scale=2.5)
g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
plt.legend (bbox_to_anchor= (1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Classification Metric')
plt.savefig('./benchmark_models_performance.png',dpi=300)

pickle.dump(lgr, open('CKD.pkl','wb'))