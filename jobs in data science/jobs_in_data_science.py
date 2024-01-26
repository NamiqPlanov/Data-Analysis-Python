import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

data=pd.read_csv('jobs in data science/jobs_in_data.csv')
'''
print(summary_info)
print('is there any missing value?{}'.format(data.isnull().values.any()))
print('is there any duplicated value?{}'.format(data.duplicated().values.any()))

sns.histplot(x='salary',data=data,bins=14,kde=True,color='green',edgecolor='grey')
plt.title('Exploring of the distribution of salaries')
plt.xlabel('Salary')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.grid(True)
plt.show()

sns.histplot(x='salary',data=data,bins=14,kde=True,color='green',edgecolor='grey')
plt.title('Exploring of the distribution of salaries without outliers')
plt.xlabel('Salary')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.grid(True)
plt.show()
sns.boxplot(x=data['salary'],color='green')
plt.title('Exploring of the distribution of salaries with outliers')
plt.show()

print('Summary info:\n{}'.format(summary_info))
print('\nOutliers:\n{}'.format(outliers))

sns.barplot(x=info1.index,y=info1.values,color='green')
plt.title('Explore the distribution of job_title using barplot')
plt.xlabel('Job Title')
plt.xticks(fontsize=8)
plt.ylabel('Number of jobs')
plt.tight_layout()
plt.show()

sns.barplot(x=info2.index,y=info2.values,color='green')
plt.title('Explore the distribution of job category using barplot')
plt.xlabel('Job Title')
plt.xticks(fontsize=6)
plt.ylabel('Number of jobs categories')
plt.tight_layout()
plt.show()

plt.pie(info3,labels=info3.index,startangle=15,autopct='%4.2f%%')
plt.title('Distribution of employment type with pie chart')
plt.show()

sns.barplot(x='company_location',y='salary',data=sorted_info5,color='green')
plt.title('Visualizing the average salary in different company locations')
plt.xlabel('Location of the company')
plt.ylabel('Average salary')
plt.xticks(rotation=30,color='blue')
plt.yticks(color='blue')
plt.grid(True)
plt.show()

for bar in bars:
    plt.text(bar.get_width(),bar.get_y()+bar.get_height()/2,f'{bar.get_width()}',va='center',ha='left',fontsize=11,color='blue')
plt.title('Distribution of the number of each company size with horizontal barplot')
plt.xlabel('Company size')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()

sns.scatterplot(x='company_size',y='salary',data=data,alpha=0.8,color='green')
plt.title('Relationship between company size and salary')
plt.xlabel('Company size')
plt.ylabel('Salary')
plt.xticks(color='blue',fontsize=10)
plt.yticks(color='blue',fontsize=10)
plt.show()

sns.scatterplot(x='experience_level',y='salary',data=data,color='green',alpha=0.6)
plt.title('Relationship between experience level and salary')
plt.xlabel('Experience level')
plt.ylabel('Salary')
plt.show()

for bar2 in bars2:
    plt.text(bar2.get_x()+bar2.get_width()/2,bar2.get_height(),f'{int(bar2.get_height())}',va='bottom',ha='center',fontsize=11,color='black')
plt.title('Distribution of Work Setting Categories')
plt.xlabel('Work Setting')
plt.show()

sns.scatterplot(x='work_setting',y='salary',data=data,color='grey',alpha=0.6)
plt.title('Relationship between work setting and salary')
plt.show()

sns.heatmap(correleation,annot=True)
plt.title('Correleation of numerical columns')
plt.show()


for bar3 in bars3:
    plt.text(bar3.get_x()+bar3.get_width()/2,bar3.get_height,f'{bar3.get_height()}',va='bottom',ha='center',fontsize=11,color='black')
plt.plot(info9['work_year'],info9['salary'],color='green',linewidth=7,linestyle='dotted')
plt.xlabel('Work year')
plt.ylabel('Average Salary')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()
'''

summary_info = data['salary'].describe()
data=data.drop_duplicates()
Q1 = data['salary'].quantile(0.25)
Q3=data['salary'].quantile(0.75)
iqr=Q3-Q1
lower_bound = Q1-1.5*iqr
upper_bound=Q3+1.5*iqr
outliers = data[(data['salary']<lower_bound)|(data['salary']>upper_bound)]
info1 = data['job_title'].value_counts().head(8)
info2 = data['job_category'].value_counts().head(8)
info3 = data['employment_type'].value_counts()
info4 = data['company_location'].value_counts().head(5)
info5=data.groupby('company_location')['salary'].mean().reset_index()
sorted_info5 = info5.sort_values(by='salary',ascending=False).head(5)
info6 = data['company_size'].value_counts()
#bars=plt.barh(info6.index,info6.values,color='green')
info7=data['work_setting'].value_counts()
#bars2 = plt.bar(info7.index,info7.values,color='green')
numerical_columns = data.select_dtypes(include='number')
correleation = numerical_columns.corr()
info8 = data['employee_residence'].value_counts().head(10)
data['work_year']=pd.to_datetime(data['work_year'],errors='coerce')
info9=data.groupby(data['work_year'])['salary'].mean().reset_index()
info10=data.groupby('job_categories')['salary'].mean().reset_index()
sorted_info10=info10.sort_values(by='salary',ascending=False)
bars3 = plt.bar(info10['work_year'],info10['salary'],color='green')









