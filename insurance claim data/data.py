import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob


data=pd.read_csv('insurance claim data/Insurance claims data.csv')
#print('Number of registered insurances-{}'.format(data.shape[0]))
#print('Is there any missing value? {}'.format(data.isnull().values.any()))
#print('Is there any duplicated value? {}'.format(data.duplicated().values.any()))

numerical_columns=data.select_dtypes(include='number')
#print(numerical_columns.describe(include='all'))

'''
sns.histplot(data['subscription_length'],kde=True,bins=12,color='green',edgecolor='white')
plt.title('Analyzing the subscription length with histplot')
plt.xlabel('Subscription length',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.show()

sns.histplot(data['vehicle_age'],kde=True,bins=12,color='green',edgecolor='white')
plt.title('Analyzing the age of each vehicle with histplot')
plt.xlabel('Ages of various vehicles',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.show()


sns.histplot(data['length'],kde=True,bins=12,color='green',edgecolor='white')
plt.title('Analyzing the length of each vehicle with histplot')
plt.xlabel('Length of various vehicles',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.show()

sns.histplot(data['gross_weight'],kde=True,bins=12,color='green',edgecolor='white')
plt.title('Analyzing the gross weight of each vehicle with histplot')
plt.xlabel('Grossweight',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.show()
'''
'''
info1=data['region_code'].value_counts().head(10)
sns.barplot(x=info1.index,y=info1.values,color='blue')
plt.title('Visualising top 10 regions for the number of  insurances')
plt.xlabel('Codes of each regions',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.tight_layout()
plt.show()

sns.countplot(x='region_code',hue='rear_brakes_type',data=data)
plt.title('Analyzing the distribution of region codes based on the types of rear brakes')
plt.xlabel('Region codes',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.legend(loc='upper right')
plt.show()

sns.countplot(x='region_code',hue='transmission_type',data=data)
plt.title('Analyzing the distribution of region codes based on the types of transmission')
plt.xlabel('Region codes',labelpad=14)
plt.ylabel('Number of insurances',labelpad=14)
plt.legend(loc='upper right')
plt.show()
'''
'''
info2=data.groupby('fuel_type')['subscription_length'].mean().reset_index()
barplot1=plt.bar(info2['fuel_type'],info2['subscription_length'],color='grey')
for a in barplot1:
    plt.text(a.get_x()+a.get_width()/2,a.get_height(),f'{a.get_height():.2f}',va='bottom',ha='center')
plt.title('Analyzing the distribution of type of fuel of each vehicle based on the length of subscriptions')
plt.xlabel('Fuel type',labelpad=14)
plt.ylabel('Lenght of subscriptions',labelpad=14)
plt.grid()
plt.show()
'''

sns.heatmap(numerical_columns.corr(),annot=True)
plt.title('Figuring out the relationship between numerical columns')
plt.show()