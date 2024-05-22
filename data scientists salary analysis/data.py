import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob


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

info2=data.groupby('employment_type')['salary_in_usd'].mean().head(10).reset_index()
info2['salary_in_usd']=info2['salary_in_usd'].astype('int')
print('Average salary for each employment type\n{}'.format(info2))
'''
info4=data.groupby('job_title')['remote_ratio'].mean().head(15)
#print(info4)


#info3=data.groupby('company_location')['salary_in_usd'].mean().head(10).reset_index()
#info3['salary_in_usd']=info3['salary_in_usd'].astype('int')
#print('Average salary for each location of the company\n{}'.format(info3))

#info3=data.groupby('employee_residence')['salary_in_usd'].mean().head(10).reset_index()
#info3['salary_in_usd']=info3['salary_in_usd'].astype('int')
#print('Average salary for each employee residence\n{}'.format(info3))

plt.plot(data['salary_in_usd'],kde=True,bins=12,color='green',binwidth=2)
plt.title('Analyzing the distribution of salaries')
plt.show()