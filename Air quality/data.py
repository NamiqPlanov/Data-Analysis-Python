import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split

data=pd.read_csv('Air quality/Air_Quality.csv')
#print('Number of registered IDs-{}'.format(data.shape[0]))
data=data.drop(columns='Message',axis=1)
missing_values='Is there any missing value-{}'.format(data.isnull().values.any())
'''
if missing_values==True:
    print(data.isnull().sum())
else:
    print('There is no missing value!')
'''
#print(data['Data Value'].describe(include='all'))

data['Start_Date']=pd.to_datetime(data['Start_Date'],format='%d/%m/%Y',errors='coerce')
#print('Is there any duplicated value?{}'.format(data.duplicated().values.any()))



filtered_data_value=[x for x in data['Data Value'] if 5<=x<=45]
'''
sns.histplot(filtered_data_value,kde=True,bins=15,binwidth=4,color='green')
plt.gca().set_facecolor('grey')
plt.title('Analyzing the distribution of Data Value with histplot')
plt.xlabel('Data value',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.grid()
plt.show()
'''
info1=data['Name'].unique()
#print(info1)
info2=data['Name'].value_counts()
#print(info2)
'''
info3=data['Geo Type Name'].value_counts()
bar1=plt.bar(info3.index,info3.values,color='blue')
for i in bar1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='black')
plt.title('Visualisation of the total number of each Geo type name with barplot')
plt.xlabel('Geo Type Name',labelpad=14)
plt.ylabel('Total number',labelpad=14)
plt.grid()
plt.show()
'''
'''
sns.violinplot(x=filtered_data_value,color='brown')
plt.title('Visualizing the distribution of Data Value across different categories using violin plot')
plt.show()
'''
'''
info4=data.groupby('Name')['Data Value'].mean().reset_index()
print('Average value for each name:\n{}'.format(info4))
'''
'''
info5=data.groupby('Geo Place Name')['Data Value'].describe()
print(info5)
'''
'''
geo_type_dummy=pd.get_dummies(data['Geo Type Name'],prefix='Geo Type')
measure_dummy=pd.get_dummies(data['Measure'],prefix='Measure')
data_with_dummies=pd.concat([data,geo_type_dummy,measure_dummy],axis=1)
data_with_dummies.drop(['Geo Type Name', 'Measure'], axis=1, inplace=True)
#print(data_with_dummies.head(2))
'''

data['Start Month']=data['Start_Date'].dt.month
data['Start Month']=data['Start Month'].map({
    1:'January',
    2:'February',
    3:'March',
    4:'April',
    5:'May',
    6:'June',
    7:'July',
    8:'August',
    9:'September',
    10:'October',
    11:'November',
    12:'December'
})
data['Start Year']=data['Start_Date'].dt.year
print(data['Start Year'].head(10))