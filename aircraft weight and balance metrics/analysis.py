import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split



data=pd.read_csv('aircraft weight and balance metrics/aircraft dataset.csv')
#print('Information about null values:\n{}'.format(data.isnull().sum()))
'''
for column in data.columns:
    if data[column].isnull().any():
        avg_value=data[column].mean()
        data[column].fillna(avg_value)
'''

numerical_columns=data.select_dtypes(include='number')
#print(numerical_columns.describe())
'''
for i in numerical_columns:
    sns.histplot(x=data[i],bins=20,kde=True,color='green')
    plt.title('Analysing the distribution of {} with histplot'.format(i))
    plt.xlabel(i,labelpad=14)
    plt.ylabel('Count',labelpad=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.show()
'''
'''
number_of_plots=len(numerical_columns)
number_of_columns=3
number_of_rows=(number_of_plots//number_of_columns)+(1 if number_of_plots% number_of_columns!=0 else 0)
fig,axes=plt.subplots(number_of_rows,number_of_columns)
for i,column in enumerate(numerical_columns):
    sns.histplot(x=data[i],kde=True,ax=axes[i])
    axes[i].set_title('Distribution of {}'.format(column))

plt.tight_layout()
plt.show()
'''
'''
sns.scatterplot(x='Empty Weight (kg)',y='Maximum Takeoff Weight (kg)',data=data,alpha=0.7,color='blue')
plt.title('Analysing the relationship between Empty and Maximum Takeoff weights')
plt.xlabel('Empty Weight (kg)',labelpad=14)
plt.ylabel('Maximum Takeoff Weight (kg)',labelpad=14)
plt.show()
'''
'''
sns.histplot(x=data['Maximum Altitude (ft)'],bins=15,kde=True,fill=False,color='violet')
plt.title('Visualising the distribution of Maximum altitude of aircrafts')
plt.show()
'''
'''
info1=data['Engine Type'].value_counts()
barplot1=plt.bar(info1.index,info1.values,color='grey')
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center')
plt.title('Visualising the count of each engine type with barplot')
plt.xlabel('Engine types',labelpad=14)
plt.ylabel('Number of each engine type',labelpad=14)
plt.xticks(rotation=45,fontsize=11,color='green')
plt.yticks(fontsize=11,color='green')
plt.show()
'''
data['Maximum Takeoff/Empty weight ratio']=data['Maximum Takeoff Weight (kg)']//data['Empty Weight (kg)']
#print(data['Maximum Takeoff/Empty weight ratio'].head(10))
data['Total Weight Capacity']=data['Empty Weight (kg)']+data['Fuel Capacity (liters)']+data['Cargo Capacity (kg)']
#print(data['Total Weight Capacity'].head(10))
'''
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True,cmap='coolwarm',linewidths=.6)
plt.title('Visualising the correlation matrix between numerical columns with heatmap')
plt.show()
'''
'''
sns.scatterplot(x='Maximum Speed (knots)',y='Number of Passengers',data=data,alpha=0.9,color='violet')
plt.title('Exploring the relationship between the Number of Passengers and Maximum Speed using a scatter plot')
plt.xlabel('Maximum Speed (knots)',labelpad=14)
plt.ylabel('Number of Passengers',labelpad=14)
plt.xticks(fontsize=11,color='grey')
plt.yticks(fontsize=11,color='grey')
plt.show()
'''
'''
sns.scatterplot(x='Wing Loading (kg/m²)',y='Tail Loading (kg/m²)',data=data,alpha=0.9,color='violet')
plt.title('Exploring the relationship between the Wing Loading  and Tail Loading  using a scatter plot')
plt.xlabel('Wing Loading',labelpad=14)
plt.ylabel('Tail Loading',labelpad=14)
plt.xticks(fontsize=11,color='grey')
plt.yticks(fontsize=11,color='grey')
plt.show()
'''
info2=data.groupby('Engine Type')['Maximum Altitude (ft)'].mean().reset_index()
sorted_info2=info2.sort_values(by='Maximum Altitude (ft)',ascending=False)
barplot2=plt.barh(sorted_info2['Engine Type'],sorted_info2['Maximum Altitude (ft)'],color='blue')
for bar in barplot2:
    plt.text(bar.get_width(),bar.get_y()+bar.get_height()/2,f'{bar.get_width()}',va='center',ha='left')
plt.title('Visualising the average maximum altitude for each engine type')
plt.show()