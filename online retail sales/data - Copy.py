import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from scipy.stats import ttest_ind,f_oneway
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import ttest_ind,f_oneway

data=pd.read_csv('online retail sales/Online_retail_sales.csv')
#print('Number of orders before removing null and duplicated values-{}'.format(data.shape[0]))
#print('Is there any missing value? {}'.format(data.isnull().values.any()))

data['credit_score']=data['credit_score'].fillna(513)
data[' monthly_income ']=data[' monthly_income '].fillna('$38,715')
#data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data[' Cost ']=data[' Cost '].apply(lambda x: int(str(x).replace('$','').replace(',','').strip()))
data[' Price ']=data[' Price '].apply(lambda x:int(str(x).replace('$','').replace(',','').strip()))
data['cart_addition_time']=pd.to_datetime(data['cart_addition_time'],errors='coerce')
data['cart_addition_time']=data['cart_addition_time'].fillna('2021-10-23 19:37:00')
data['order_return']=data['order_return'].fillna('TRUE')
#print(data['cart_addition_time'].head(15))
numerical_cols=data.select_dtypes(include='number')
#print(numerical_cols.describe())

#print(numerical_cols.corr())
#sns.heatmap(numerical_cols.corr(),cmap='flare',annot=True)
#plt.title('Distribution the relationship between numerical columns with heatmap')
#plt.show()

info1=data['gender'].value_counts()
'''
barplot1=plt.bar(info1.index,info1.values,color='green')
for bar1 in barplot1:
    plt.text(bar1.get_x()+bar1.get_width()/2,bar1.get_height(),f'{bar1.get_height()}',va='bottom',ha='center')
plt.title('Figuring out the total number of each gender')
plt.xlabel('Gender',labelpad=14)
plt.ylabel('Total number',labelpad=14)
plt.grid()
plt.show()


countplot1=sns.countplot(x='country',hue='gender',data=data,color='blue')
for a in countplot1.containers:
    countplot1.bar_label(a)
plt.title('Analyzing the distribution of sales for each country')
plt.xlabel('Country',labelpad=14)
plt.ylabel('Total number of online sales',labelpad=14)
plt.legend(loc='upper right')
plt.show()

countplot2=sns.countplot(x='Category',hue='order_confirmation',data=data,color='blue')
for a in countplot2.containers:
    countplot2.bar_label(a)
plt.title('Analyzing the distribution of sales for each category')
plt.xlabel('Category',labelpad=14)
plt.ylabel('Total number of online sales',labelpad=14)
plt.legend(loc='upper right')
plt.show()

'''
'''
sns.histplot(data['credit_score'],kde=True,binwidth=2,bins=13,color='yellow')
plt.title('Visualizing the distribution of credit score with histplot')
plt.xlabel('Scores',labelpad=14)
plt.ylabel('Number of sales',labelpad=14)
plt.show()

sns.histplot(data[' monthly_income '],kde=True,binwidth=2,bins=13,color='yellow')
plt.title('Visualizing the distribution of salaries with histplot')
plt.xlabel('Salary',labelpad=14)
plt.ylabel('Number of sales',labelpad=14)
plt.show()

plt.plot(data.index,data[' Cost '],label='Cost',marker='o',color='blue')
plt.plot(data.index,data[' Price '],label='Price',marker='o',color='green')
plt.title('Showing the difference between Cost and Price of sales (Cost-blue,Price-green)')
plt.show()
'''
data['Card addition month']=data['cart_addition_time'].dt.month
data['Card addition year']=data['cart_addition_time'].dt.year
data['Card addition day']=data['cart_addition_time'].dt.day

data['Card addition month']=data['Card addition month'].map({
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
'''
info2=data.groupby('Card addition year').size()
info2.plot(kind='line',marker='o',color='grey')
plt.title('Analyzing of total sales over year')
plt.xlabel('Year',labelpad=14)
plt.ylabel('Total number of sales',labelpad=14)
plt.show()

info2=data.groupby('Card addition month').size()
info2.plot(kind='line',marker='o',color='grey')
plt.title('Analyzing of total sales over monthes')
plt.xlabel('Month',labelpad=14)
plt.ylabel('Total number of sales',labelpad=14)
plt.show()
'''

