import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler,StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind


data=pd.read_csv('laptop analysis/best_buy_laptops_2024.csv')

for i in range(len(data['offers/price'])):
    if i%2==0:
        if pd.isnull(data.at[i,'offers/price']):
            data.at[i,'offers/price']=959
    else:
        if pd.isnull(data.at[i,'offers/price']):
            data.at[i,'offers/price']=1240

data['aggregateRating/ratingValue'].fillna(data['aggregateRating/ratingValue'].mean(),inplace=True)
for i in range(len(data['aggregateRating/reviewCount'])):
    if i%2==0:
        if pd.isnull(data.at[i,'aggregateRating/reviewCount']):
            data.at[i,'aggregateRating/reviewCount']=650
    else:
        if pd.isnull(data.at[i,'aggregateRating/reviewCount']):
            data.at[i,'aggregateRating/reviewCount']=730
data['depth'].fillna(data['depth'].mean(),inplace=True)
data['width'].fillna(data['width'].mean(),inplace=True)
data.dropna(inplace=True)
#print('Is there any missing (null) value after dropping and filling missing values with specific values?{}'.format(data.isnull().values.any()))
numerical_columns=data.select_dtypes(include='number')
#print(numerical_columns.describe())

'''for i in numerical_columns:
    sns.histplot(data[i],kde=True,color='green')
    plt.title('Histplot of {}'.format(data[i]))
    plt.xlabel(data[i],labelpad=14)
    plt.ylabel('Count',labelpad=14)
    plt.show()
    '''
categorical_colums=['brand','model','offers/price']
'''
info1=data['brand'].value_counts().head(10)
sns.barplot(x=info1.index,y=info1.values,fill=False,color='grey')
plt.title('Analyzing the distribution of top laptop brands')
plt.xlabel('Laptops',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()
info2=data['model'].value_counts().head(10)
barplot2=plt.bar(info2.index,info2.values,fill=True,color='green')
for k in barplot2:
    plt.text(k.get_x()+k.get_width()/2,k.get_height(),f'{k.get_height()}',va='bottom',ha='center')
plt.title('Analyzing the distribution of top laptop models')
plt.xlabel('Laptop models',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()
info3=data['offers/price'].value_counts().head(10)
sns.barplot(x=info3.index,y=info3.values,fill=False,color='blue')
plt.title('Analyzing the distribution of offers/price')
plt.xlabel('',labelpad=14)
plt.ylabel('',labelpad=14)
plt.show()
'''
'''
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True,cmap='coolwarm')
plt.title('Analyzing correlations between numerical variables with heatmap')
plt.show()
'''
'''
crosstab=pd.crosstab(data['brand'],data['offers/price'])
crosstab.plot(kind='bar',stacked=False)
plt.title('Analyzing relationship between brand and offers/price')
plt.tight_layout()
plt.show()
'''
sns.scatterplot(x='depth',y='width',data=data,alpha=0.7,color='green')
plt.title('Visualisation the relationship between depth and width of laptops')
plt.xlabel('Depth',labelpad=13)
plt.ylabel('Width',labelpad=13)
plt.xticks(color='blue')
plt.yticks(color='blue')
plt.show()
    
                 
            