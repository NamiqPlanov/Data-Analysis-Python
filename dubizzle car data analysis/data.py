import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv('dubizzle car data analysis/dubizzle_cars_dataset.csv')
#print('Number of registered cars null values-{}'.format(data.shape[0]))
data['trim']=data['trim'].fillna('Luxury')
data['engine_capacity_cc']=data['engine_capacity_cc'].fillna('2500 - 2999 cc')
data['horsepower']=data['horsepower'].fillna('300-377 HP')
data['area_name']=data['area_name'].fillna('Al Mamzar')
data['location_name']=data['location_name'].fillna('Al Mankhool')
data['latitude']=data['latitude'].fillna(data['latitude'].mean())
data['longitude']=data['longitude'].fillna(data['longitude'].mean())
#print('Is there any missing value?{}'.format(data.isnull().values.any()))
numerical_cols=data.select_dtypes(include='number')
#print(numerical_cols.describe())
#print('Is there any duplicated value? {}'.format(data.duplicated().values.any()))
data.drop_duplicates(inplace=True)
#print('Is there any duplicated valueafter removing them? {}'.format(data.duplicated().values.any()))
'''
print(numerical_cols.corr())
sns.heatmap(numerical_cols.corr(),annot=True)
plt.title('Distribution the relationship between numerical columns')
plt.show()
'''
'''
info1=data['brand'].value_counts().head(15)
barplot1=plt.bar(info1.index,info1.values,color='green')
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center')
plt.title('Visualizing top 15 car brands for manufacturing with barplot')
plt.xlabel('Brands',labelpad=14)
plt.ylabel('Total number of cars',labelpad=14)
plt.show()


info2=sns.countplot(x='vehicle_age_years',data=data,color='green')
for a in info2.containers:
    info2.bar_label(a)
plt.title('Grouping the cars based on their ages')
plt.xlabel('Age of cars',labelpad=14)
plt.ylabel('Number of cars',labelpad=14)
plt.show()
'''
info3=data['body_type'].value_counts()
print('Number of cars based on their body types:\n{}'.format(info3))