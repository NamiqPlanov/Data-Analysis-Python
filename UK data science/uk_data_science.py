import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.linear_model import LinearRegression



data=pd.read_csv('Uk data science/UK_data_science.csv')
'''
print('is there any missing value?{}'.format(data.isnull().values.any()))
print('is there any duplicated value?{}'.format(data.duplicated().values.any()))

print('number of rows-{}'.format(data.shape[0]))
print('number of rows-{}'.format(data.shape[1]))

sns.barplot(x='company',y='salary',data=sorted_info1,color='green')
plt.title('Counting the average salary for each company')
plt.xlabel('Company')
plt.ylabel('Average Salary')
plt.xticks(rotation=45,color='blue',fontsize=10)
plt.grid(True)
plt.show()

sns.heatmap(numerical_columns.corr(),annot=True)
plt.title('Distribution of correlation matrix of numerical columns with heatmap')
plt.show()

for bar in info2:
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{bar.get_height()}',color='blue',va='bottom',ha='center',fontsize=11)
plt.title('Exploring the distribution of ratings')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

for i in arr:
    Q1=data[i].quantile(0.25)
    Q3=data[i].quantile(0.75)
    iqr=Q3-Q1
    lower_bound=Q1-1.5*iqr
    upper_bound=Q3+1.5*iqr
    outliers = data[(data[i]<lower_bound)|(data[i]>upper_bound)]
    sns.histplot(x=i,bins=10,color='green',edgecolor='grey',kde=True,data=data)
    plt.title('Distribution of {} without outliers'.format(data[i]))
    plt.xlabel(i)
    plt.xticks(color='grey',fontsize=10)
    plt.grid(True)
    plt.show()
    sns.boxplot(x=data[i],color='green')
    plt.title('Distribution of {} with outliers'.format(data[i]))
    plt.xlabel(i)
    plt.xticks(color='grey',fontsize=10)
    plt.show()
    summary_info=data[i].describe()
    print('Summary info about column:\n{}'.format(summary_info))
    print('\nOutliers:{}'.format(outliers))
    

sns.barplot(x=info3.index,y=info3.values,color='green')
plt.title('Exploring the distribution of reviews count with barplot')
plt.xlabel('Reviews Count')
plt.show()

sns.barplot(x=info4.index,y=info4.values,color='green')
plt.title('Exploring top companies for the number of job positions')
plt.xlabel('Company')
plt.ylabel('Job Applications')
plt.show()


print('The average value for rating is {}'.format(data['rating'].mean()))
print('The median value for rating is {}'.format(data['rating'].median()))
print('The standard deviation  for rating is {}'.format(data['rating'].std()))
print('-----------------------------------------------------------------')
print('The average value for reviews count is {}'.format(data['reviewsCount'].mean()))
print('The median value for reviews count is {}'.format(data['reviewsCount'].median()))
print('The standard deviation  for reviews count is {}'.format(data['reviewsCount'].std()))


print('Companies with the lowest average review count:\n{}'.format(min_info5))
print('\nCompanies with the lowest average review count:\n{}'.format(max_info5))


for i in job_types_info.containers:
    job_types_info.bar_label(i)
plt.title('Exploring the distribution of types of jobs')
plt.xlabel('Job types')
plt.ylabel('Count')
plt.xticks(rotation=60,color='blue',fontsize=9)
plt.yticks(color='blue',fontsize=8)
plt.show()

print('Top job positions based on rating:\n{}'.format(sorted_info8))
'''

data=data.dropna()
data=data.drop_duplicates()
data['salary']=pd.to_numeric(data['salary'],errors='coerce')
info1 = data.groupby('company')['salary'].mean().reset_index()
sorted_info1=info1.sort_values(by='salary',ascending=False).head(10)
numerical_columns=data.select_dtypes(include='number')
data=data.drop('salary',axis=1)
info_rating=data['rating'].value_counts().head(10)
#info2=plt.bar(info_rating.index,info_rating.values,color='green')
arr = ['rating','reviewsCount']
info3=data['reviewsCount'].value_counts().head(5)
info4=data['company'].value_counts().head(5)
info5=data.groupby('company')['reviewsCount'].mean().reset_index()
min_info5=info5.sort_values(by='reviewsCount',ascending=False).tail(5)
max_info5=info5.sort_values(by='reviewsCount',ascending=False).head(5)
#job_types_info=sns.countplot(x='jobTypeConsolidated',data=data,color='green')
def consolidate_job(position):
    if 'data scientist' in position.lower():
        return 'Data scientist'
    elif 'engineer' in position.lower():
        return 'Engineer'
    elif 'analyst' in position.lower():
        return 'Analyst'
    else:
        return 'Other'
data['Consolidated job type']=data['positionName'].apply(consolidate_job)
info7=data['Consolidated job type'].value_counts()
info8=data.groupby('positionName')['rating'].mean().reset_index()
sorted_info8=info8.sort_values(by='rating',ascending=False).head(5)
data['rating']=pd.to_numeric(data['rating'],errors='coerce')
fulltime=data[data['jobTypeConsolidated']=='Full-Time']['rating']
parttime=data[data['jobTypeConsolidated']=='Part-Time']['rating']
plt.hist(fulltime,bins=25,color='green',label='Full Time',alpha=0.7)
plt.hist(parttime,bins=15,color='orange',label='Part Time',alpha=0.7)
plt.legend()
plt.show()



