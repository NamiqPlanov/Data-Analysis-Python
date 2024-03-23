import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

data=pd.read_csv('1000 companies and their statistics/1000 companies list.csv')


data.columns=[col.capitalize() for col in data.columns]
#print('Is there missing values-{}'.format(data.isnull().values.any()))
data['Type']=data['Type'].fillna('Cyber Security')
data['Age']=data['Age'].fillna('56 years old')
data['Age']=data['Age'].str.extract('(\d+)').astype(float)
data['Reviewers']=data['Reviewers'].str.extract('(\d+)').astype(float)
data=data.drop(columns='Critically rated for',axis=1)
data['Highly rated for']=data['Highly rated for'].fillna('Job Security,Skill Development,Work Satisfaction')
#print(data.isnull().sum())
#print(data['Rating'].describe())
'''
sns.set_style('darkgrid')
sns.histplot(data['Rating'],kde=True,binwidth=4,bins=15,color='blue')
plt.gca().set_facecolor('grey')
plt.gcf().set_facecolor('black')
plt.title('Determining the distribution of ratings',color='white')
plt.xlabel('Ratings',color='white',labelpad=14)
plt.ylabel('Count',color='white',labelpad=14)
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()
'''

info1=data['Type'].value_counts().head(10)
'''
sns.barplot(x=info1.index,y=info1.values,color='green')
plt.gcf().set_facecolor('#D2691E')
plt.title('Analyzing the top 10 Type of compamies',color='white')
plt.xlabel('Type of Company',color='white',labelpad=14)
plt.ylabel('Number of each type of company',labelpad=14,color='white')
plt.xticks(fontsize=9,color='white')
plt.yticks(color='white')
plt.show()
'''
'''
sns.violinplot(x='Rating',data=data,color='green')
plt.title('Visualising the distribution of Rating with viloin plot')
plt.xlabel('Rating',labelpad=14)
plt.ylabel('',labelpad=14)
plt.show()
'''
'''
sns.histplot(data['Age'],kde=True,binwidth=4,bins=15,color='blue')
plt.gcf().set_facecolor('grey')
plt.title('Determining the distribution of ratings',color='black')
plt.xlabel('Ratings',color='black',labelpad=14)
plt.ylabel('Count',color='black',labelpad=14)
plt.xticks(color='black')
plt.yticks(color='black')
plt.show()
'''
'''
info3=data.groupby('Type')['Rating'].mean().reset_index()
sorted_info3=info3.sort_values(by='Rating',ascending=False).head(10)
print(sorted_info3)
'''
'''
info4=data.groupby('Type')['Reviewers'].sum().reset_index()
sorted_info4=info4.sort_values(by='Reviewers',ascending=False).head(10)
print(sorted_info4)
'''
numerical_columns=data.select_dtypes(include='number')
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True)
plt.title('Visualising the correlation matrix between numerical columns with heatmap')
plt.show()