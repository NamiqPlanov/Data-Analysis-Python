import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error


data=pd.read_csv('car sales analysis 2/car data.csv')
#print('Number of registered cars-{}'.format(data.shape[0]))
#print('Is there any missing value? {}'.format(data.isnull().values.any()))
numerical_columns=data.select_dtypes(include='number')
#print(numerical_columns.describe())

'''   
sns.histplot(data['car_engine_capacity'],kde=True,bins=20,binwidth=3,color='green')
plt.title('Analyzing the distribution of car engine capacity with histplot')
plt.show()
'''
'''
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True)
plt.title('Visualisation the relationship between numerical columns')
plt.show()
'''
info1=data['car_brand'].value_counts().head(15)
#print(info1)
'''
sns.barplot(x=info1.index,y=info1.values,color='blue')
plt.title('Exploring the total number of cars of each car brand')
plt.xlabel('Car brand',labelpad=14)
plt.ylabel('Total number of cars',labelpad=14)
plt.xticks(fontsize=7)
plt.grid()
plt.show()
'''
'''
info2=sns.countplot(x='car_transmission',data=data,hue='car_fuel')
for i in info2.containers:
    info2.bar_label(i)
plt.title('Exploring the numbers of car transmissions for each car fuel')
plt.xlabel('Car transmission',labelpad=14)
plt.ylabel('Total number of cars',labelpad=14)
plt.legend(loc='upper right')
plt.show()
'''
'''
info3=data.groupby('car_brand')['car_price'].mean().reset_index()
sorted_info3=info3.head(10)
sns.barplot(x='car_brand',y='car_price',data=sorted_info3,color='brown')
plt.title('Comparing average car prices across different brands')
plt.xlabel('Car brand',labelpad=15)
plt.ylabel('Car price',labelpad=15)
plt.grid()
plt.show()


info4=data.groupby('car_city')['car_price'].mean().reset_index()
sorted_info4=info4.head(10)
sns.barplot(x='car_city',y='car_price',data=sorted_info4,color='brown')
plt.title('Comparing average car prices across different cities')
plt.xlabel('City',labelpad=15)
plt.ylabel('Car price',labelpad=15)
plt.grid()
plt.show()
'''

data['Manufacturing year']=2024-data['car_age']
#print(data['Manufacturing year'].head(10))
'''
categorical_columns=['car_brand','car_city','car_transmission']
onehot=OneHotEncoder()
onehot_data=pd.DataFrame(onehot.fit_transform(data[categorical_columns]))
onehot_data_normalized=pd.concat([data.drop(columns=categorical_columns),onehot_data],axis=1)
print(onehot_data_normalized)
'''
'''
sns.violinplot(x=data['car_price'],color='blue')
plt.title('Visualizing the distribution of car prices across different categories using violin plot')
plt.show()
'''

sns.scatterplot(x='car_mileage',y='car_engine_capacity',data=data,color='grey',alpha=0.7)
plt.title('Relationship between car mileage and engine capacity')
plt.show()

sns.scatterplot(x='car_mileage',y='car_engine_hp',data=data,color='grey',alpha=0.7)
plt.title('Relationship between car mileage and engine power')
plt.show()

sns.scatterplot(x='car_engine_hp',y='car_engine_capacity',data=data,color='grey',alpha=0.7)
plt.title('Relationship between engine power and engine capacity')
plt.show()