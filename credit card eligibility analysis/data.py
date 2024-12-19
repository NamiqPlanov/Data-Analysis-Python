import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from scipy.stats import ttest_ind,f_oneway
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import ttest_ind,f_oneway


data=pd.read_csv('credit card eligibility analysis/dataset.csv',)
data=data.drop(columns='Target',axis=1)
data['Gender']=data['Gender'].apply(lambda x:'Male' if x==1 else 'Female')
data['Own_car']=data['Own_car'].apply(lambda x:'Yes' if x==1 else 'No')
data['Own_property']=data['Own_property'].apply(lambda x:'Provided' if x==1 else 'Not provided')
data['Work_phone']=data['Work_phone'].apply(lambda x:'Provided' if x==1 else 'Not provided')
data['Phone']=data['Phone'].apply(lambda x:'Contains' if x==1 else 'Does not contain')
data['Email']=data['Email'].apply(lambda x:'Provided' if x==1 else 'Not provided')
data['Unemployed']=data['Unemployed'].apply(lambda x:'True' if x==1 else 'False')
data['Age']=data['Age'].astype('int')
data['Years_employed']=data['Years_employed'].astype('int')
print('Is there any missing value?-{}'.format(data.isnull().values.any()))
print('Is there any duplicated value?-{}'.format(data.duplicated().values.any()))
numerical_columns=data.select_dtypes(include='number')
print(numerical_columns.describe(include='all'))
sns.histplot(data['Age'],kde=True,bins=12,binwidth=2,color='violet')
plt.title('Visualising the distribution of age with histplot')
plt.xlabel('Age',labelpad=14)
plt.ylabel('Total number of card users',labelpad=14)
plt.grid()
plt.show()

sns.histplot(data['Num_children'],kde=True,bins=12,binwidth=2,color='grey')
plt.title('Visualising the distribution of Number of children with histplot')
plt.xlabel('Number of children',labelpad=14)
plt.ylabel('',labelpad=14)
plt.grid()
plt.show()

sns.histplot(data['Account_length'],kde=True,bins=12,binwidth=2,color='green')
plt.title('Visualising the distribution of length of each account with histplot')
plt.xlabel('Length of account',labelpad=14)
plt.ylabel('Total number of accounts',labelpad=14)
plt.grid()
plt.show()

sns.histplot(data['Total_income'],kde=True,bins=12,binwidth=2,color='blue')
plt.title('Visualising the distribution of total income with histplot')
plt.xlabel('Income',labelpad=14)
plt.ylabel('Number of accounts',labelpad=14)
plt.grid()
plt.show()

sns.histplot(data['Years_employed'],kde=True,bins=12,binwidth=2,color='yellow')
plt.title('Visualising the distribution of years of being employed with histplot')
plt.xlabel('Years of being employed',labelpad=14)
plt.ylabel('Number of accounts',labelpad=14)
plt.grid()
plt.show()

info1=data['Income_type'].value_counts()
plot1=plt.bar(info1.index,info1.values,color='green')
for i in plot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center')
plt.title('Analysing the distribution of the types of income')
plt.xlabel('Type of income',labelpad=14)
plt.ylabel('Total number of accounts',labelpad=14)
plt.show()
countplot1=sns.countplot(x='Education_type',hue='Own_car',data=data,color='grey')
for i in countplot1.containers:
    countplot1.bar_label(i)
plt.title('Analysing the distribution of the types of education')
plt.xlabel('Type of education',labelpad=14)
plt.ylabel('Total number of accounts',labelpad=14)
plt.legend(loc='upper right')
plt.show()
print(numerical_columns.corr())
categorical_columns=data[['Gender','Income_type','Family_status']]


ages=[20,30,40,50,60,70]
age_groups=['20-30','31-40','41-50','51-60','61-70']
data['Age groups']=pd.cut(data['Age'],bins=ages,labels=age_groups,right=False)



arr1=data[['Years_employed','Income_type','Years_employed']]
target1=data['Total_income']
arr1_normalized=pd.get_dummies(arr1,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(arr1_normalized,target1,random_state=42,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)

sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual Income')
sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted Income')
plt.show()

