import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from scipy.stats import ttest_ind,f_oneway


data=pd.read_csv('data scientists salary analysis/salaries _2.csv')
'''
if data.isnull().values.any()==True:
    print('There are missing values')
else:
    print('There is not any missing value')
'''
numerical_columns=data.select_dtypes(include='number')
#print(numerical_columns.describe())

#print(data['experience_level'].unique())
#print(data['employment_type'].unique())
#print(data['job_title'].unique())
#print(data['employee_residence'].unique())
#print(data['company_location'].unique())

data['difference in salaries (usd and other currencies)']=data['salary_in_usd']-data['salary']
#print(data['difference in salaries (usd and other currencies)'].head(10))

'''
info1=data.groupby('job_title')['salary_in_usd'].mean().head(10).reset_index()
info1['salary_in_usd']=info1['salary_in_usd'].astype('int')
print('Average salary for each job\n{}'.format(info1))
print('------------------------------------------------------------')

info2=data.groupby('experience_level')['salary_in_usd'].mean().head(10).reset_index()
info2['salary_in_usd']=info2['salary_in_usd'].astype('int')
print('Average salary for each experience level\n{}'.format(info2))
print('------------------------------------------------------------')

info3=data.groupby('employment_type')['salary_in_usd'].mean().head(10).reset_index()
info3['salary_in_usd']=info3['salary_in_usd'].astype('int')
print('Average salary for each employment type\n{}'.format(info3))
'''
info4=data.groupby('job_title')['remote_ratio'].mean().head(15)
#print(info4)


#info5=data.groupby('company_location')['salary_in_usd'].mean().head(10).reset_index()
#info5['salary_in_usd']=info5['salary_in_usd'].astype('int')
#print('Average salary for each location of the company\n{}'.format(info5))

#info6=data.groupby('employee_residence')['salary_in_usd'].mean().head(10).reset_index()
#info6['salary_in_usd']=info6['salary_in_usd'].astype('int')
#print('Average salary for each employee residence\n{}'.format(info6))

#print(numerical_columns.corr())
#sns.heatmap(numerical_columns.corr(),annot=True)
#plt.title('Discovering the relationship between numerical values with heatmap')
#plt.show()

#sns.boxplot(x='experience_level',y='salary_in_usd',data=data,color='green')
#plt.title('Analyzing the salary distribution across different experience levels with boxplot')
#plt.xlabel('Experience level',labelpad=14)
#plt.ylabel('Salary in USD',labelpad=14)
#plt.show()

#sns.histplot(data['remote_ratio'],kde=True,bins=12,binwidth=2)
#plt.title('Visualizing the distribution of remote ratio with histplot')
#plt.xlabel('Remote ratio',labelpad=14)
#plt.ylabel('')
#plt.show()

#info7=data['employment_type'].value_counts()
#plt.pie(info7,labels=info7.index,autopct='%2.1f%%')
#plt.title('Creating a pie chart of employment types')
#plt.show()

data['remote_ratio_category']=data['remote_ratio'].apply(lambda x: 'No Remote' if x==0 else ('Hybrid' if x==50 else 'Remote'))
info8=data.groupby('remote_ratio_category')['salary_in_usd'].mean().reset_index()
#sns.barplot(x='remote_ratio_category',y='salary_in_usd',data=info8,color='green')
#plt.title('Comparing the average salary for each remote ratio category')
#plt.xlabel('Remote Ratio Category',labelpad=14)
#plt.ylabel('Average salary in usd',labelpad=14)
#plt.show()

info9=data.groupby('company_size')['salary_in_usd'].mean().reset_index()
#sns.barplot(x='company_size',y='salary_in_usd',data=info9,color='green')
#plt.title('Comparing the average salary for each company size')
#plt.xlabel('Company size',labelpad=14)
#plt.ylabel('Average salary in usd',labelpad=14)
#plt.show()

arr1=data[['experience_level','company_size']]
target=data['salary_in_usd']
arr1_normalized=pd.get_dummies(arr1,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(arr1_normalized,target,random_state=42,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)

#sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual Salary in USD')
#sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted Salary in USD')
#plt.title('Actual Salary vs Predicted Salary')
#plt.show()

info10=data.groupby('work_year')['salary_in_usd'].mean().reset_index()
#plt.plot(info10['work_year'],info10['salary_in_usd'],marker='o',color='blue',linestyle='-')
#plt.title('Performing a time series analysis in salaries')
#plt.xlabel('Years',labelpad=14)
#plt.ylabel('Average salary in USD',labelpad=14)
#plt.show()


info11=data['experience_level'].unique()
salary1=[data[data['experience_level']==level]['salary'] for level in info11]
ttest,p_value=f_oneway(*salary1)
alpha=0.05
if p_value<alpha:
    print('There is significant difference in salary between various experience levels')
else:
    print('There is no significant difference in salary between various experience levels')
