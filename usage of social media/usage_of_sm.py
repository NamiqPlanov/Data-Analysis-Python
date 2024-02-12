import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder,LabelEncoder


data=pd.read_csv('usage of social media/dummy_data.csv')
'''
print('number of rows-{}'.format(data.shape[0]))
print('number of columns-{}'.format(data.shape[1]))
print('is there any missing value?{}'.format(data.isnull().values.any()))
'''
numerical_columns = data.select_dtypes(include='number')
#print(numerical_columns.describe())

info1 = data['gender'].value_counts()
info2 = data['interests'].value_counts()
info3=data['location'].value_counts()
info4=data['demographics'].value_counts()
info5=data['platform'].value_counts()
'''
print(info1)
print(info2)
print(info3)
print(info4)
'''
'''
sns.histplot(x=data['age'],kde=True,bins=12,color='blue',binwidth=3,fill=False)
plt.title('Exploring the  distribution of age of social media users')
plt.xlabel('Age')
plt.ylabel('Time spent')
plt.show()

sns.histplot(x=data['time_spent'],kde=True,bins=12,color='blue',binwidth=3,fill=True)
plt.title('Exploring the  distribution of spent time in social media ')
plt.xlabel('Time spent')
plt.show()
'''
'''
barplot1=plt.bar(info1.index,info1.values,color='green')
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center')
plt.title('Exploring the  distribution of gender of social media users')
plt.xlabel('Gender')
plt.ylabel('Number of each gender')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()

barplot2=plt.bar(info5.index,info5.values,color='green')
for a in barplot2:
    plt.text(a.get_x()+a.get_width()/2,a.get_height(),f'{a.get_height()}',va='bottom',ha='center')
plt.title('Exploring the  distribution of social media platforms that are used by users')
plt.xlabel('Platforms')
plt.ylabel('Number of each platform')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()

barplot3=plt.bar(info2.index,info2.values,color='green')
for b in barplot3:
    plt.text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height()}',va='bottom',ha='center')
plt.title('Exploring the  distribution of interests of social media users')
plt.xlabel('Interests')
plt.ylabel('Number of each interest')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()

barplot4=plt.barh(info3.index,info3.values,color='green')
for c in barplot4:
    plt.text(c.get_width(),c.get_y()+c.get_height()/2,f'{c.get_width()}',va='center',ha='left')
plt.title('Exploring the  distribution of location social media users')
plt.ylabel('Location')
plt.xlabel('Number of each location')
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()
'''
'''
plt.pie(info1,labels=info1.index,startangle=14,autopct='%5.2f%%')
plt.title('Proportion of genders')
plt.show()

plt.pie(info5,labels=info5.index,startangle=14,autopct='%5.2f%%')
plt.title('Proportion of platforms')
plt.show()
'''
'''
info6=data.groupby('gender')['time_spent'].mean().reset_index()
sorted_info6=info6.sort_values(by='time_spent',ascending=False)
sns.barplot(x='gender',y='time_spent',data=sorted_info6,color='grey',fill=False)
plt.title('Calculating the average time spent on the platform by gender')
plt.xlabel('Gender')
plt.ylabel('Average time spent')
plt.tight_layout()
plt.show()
'''
'''
info7 = sns.countplot(x='interests',hue='demographics',data=data,width=0.3)
for d in info7.containers:
    info7.bar_label(d)
plt.title('Distribution of interests across different demographics')
plt.legend(loc='upper right')
plt.show()
'''
'''
homewoners_income=data[data['isHomeOwner']==True]['income']
non_homewoners_income=data[data['isHomeOwner']==False]['income']
print('Homeowners income Summary info')
print('Mean income:{}'.format(homewoners_income.mean()))
print('Mediaan income:{}'.format(homewoners_income.median()))
print('Standart deviation of income:{}'.format(homewoners_income.std()))
print('------------------------------------------------------------')
print('Nonhomeowners income Summary info')
print('Mean income:{}'.format(non_homewoners_income.mean()))
print('Mediaan income:{}'.format(non_homewoners_income.median()))
print('Standart deviation of income:{}'.format(non_homewoners_income.std()))
'''
'''
sns.scatterplot(x='age',y='time_spent',data=data,alpha=0.7,color='green')
plt.title('Exploring the relationship between age and time spent by users on platform with scatterplot')
plt.show()

age_group=[0,20,30,40,50,60,70,90]
data['age group']=pd.cut(data['age'],bins=age_group)
info8 = data.groupby('age group')['time_spent'].mean().reset_index()
sns.barplot(x='age group',y='time_spent',data=info8,color='green')
plt.title('Exploring the relationship between age and time spent by users on platform with barplot')
plt.xlabel('Age group')
plt.ylabel('Average time spent')
plt.show()
'''
'''
info9=data['profession'].value_counts().head(1)
print('The most popular profession among users is {}'.format(info9.index))
'''
print(data['interests'].unique())
'''
age_group=[0,30,40,50,60,float('inf')]
age_group_str=['0-30','30-40','40-50','50-60','60+']
data['age group']=pd.cut(data['age'],bins=age_group,labels=age_group_str,right=False)
info10=data['age group'].value_counts()
sns.barplot(x=info10.index,y=info10.values,color='green',fill=False)
plt.title('Distribution of age group of users')
plt.xlabel('Age group',labelpad=12)
plt.ylabel('Number of users',labelpad=12)
plt.show()
'''
'''
unique_interests = data['interests'].unique()
segment={}
for interest in unique_interests:
    segment[interest]=data[data['interests']==interest]
for interest,segment_data in segment.items():
    print('Interests:')
    print(segment_data['gender'].value_counts())
    print(segment_data['age'].describe())
    print(segment_data['profession'].value_counts())
    print('Average time spent:{}'.format(segment_data['time_spent'].mean()))

sns.histplot(data=segment_data,x='time_spent',kde=True,bins=20)
plt.title(f'Distribution of Time Spent on Platform - {interest}')
plt.xlabel('Time Spent')
plt.ylabel('Frequency')
plt.show()
'''
'''
income_int=[0,10000,12000,14000,16000,18000,float('inf')]
income_str=['0-10k','10k-12k','12k-14k','14k-16k','16k-18k','18k+']
data['income level']=pd.cut(data['income'],bins=income_int,labels=income_str,right=False)

for income,segment_data in data.groupby(data['income level']):
    print(f'Income level:{income}')
    print('Average time spent:{}'.format(segment_data['time_spent'].mean()))

    sns.histplot(data=segment_data,x='time_spent',bins=20,kde=True)
    plt.title(f'Distribution of time spent on platform-{income}')
    plt.xlabel('Time spent')
    plt.ylabel('Frequency')
    plt.show()
'''
'''
x=pd.get_dummies(data.drop(columns=['time_spent']))
y=data['time_spent']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
mse=mean_squared_error(y_test,y_predict)
print('Mean Squad Error:{}'.format(mse))
'''
'''
a=pd.get_dummies(data.drop(columns=['indebt']))
b=data['indebt']
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(a_train,b_train)
b_predict=model.predict(a_test)
accuracy=accuracy_score(b_test,b_predict)
print('Accuracy:{}'.format(accuracy))
'''
categorical_columns=['gender','platform','interests','profession']
encoder=OneHotEncoder(drop='first')
encoded_columns=pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
encoded_data=pd.concat([data.drop(columns=categorical_columns),encoded_columns],axis=1)

label_encoder=LabelEncoder()
for i in categorical_columns:
    data[i]=label_encoder.fit_transform(data[i])

corr1=numerical_columns.corr()
print(corr1)
sns.heatmap(corr1,annot=True)
plt.title('Distribution the relationship between numerical columns with heatmap')
plt.show()


