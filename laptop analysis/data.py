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
from textblob import TextBlob
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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
print('Is there any missing (null) value after dropping and filling missing values with specific values?{}'.format(data.isnull().values.any()))
numerical_columns=data.select_dtypes(include='number')
print(numerical_columns.describe())

for i in numerical_columns:
    sns.histplot(data[i],kde=True,color='green')
    plt.title('Histplot of {}'.format(data[i]))
    plt.xlabel(data[i],labelpad=14)
    plt.ylabel('Count',labelpad=14)
    plt.show()
    
categorical_colums=['brand','model','offers/price']

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


print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True,cmap='coolwarm')
plt.title('Analyzing correlations between numerical variables with heatmap')
plt.show()


crosstab=pd.crosstab(data['brand'],data['offers/price'])
crosstab.plot(kind='bar',stacked=False)
plt.title('Analyzing relationship between brand and offers/price')
plt.tight_layout()
plt.show()


sns.scatterplot(x='depth',y='width',data=data,alpha=0.7,color='green')
plt.title('Visualisation the relationship between depth and width of laptops')
plt.xlabel('Depth',labelpad=13)
plt.ylabel('Width',labelpad=13)
plt.xticks(color='blue')
plt.yticks(color='blue')
plt.show()


sns.histplot(data['aggregateRating/ratingValue'],kde=True,bins=15,color='green')
plt.title('Visualizing the distribution of aggregateRating/ratingValue')
plt.xlabel('aggregateRating/ratingValue',labelpad=14)
plt.ylabel('',labelpad=14)
plt.tight_layout()
plt.show()


sns.histplot(data['offers/price'],kde=True,bins=15,color='green')
plt.title('Visualizing the distribution of offers/price')
plt.xlabel('offers/price',labelpad=14)
plt.ylabel('',labelpad=14)
plt.tight_layout()
plt.show()


info4=data.groupby('brand')['aggregateRating/ratingValue'].mean().reset_index()
sorted_info4=info4.sort_values(by='aggregateRating/ratingValue',ascending=False).head(10)
barh1=plt.barh(sorted_info4['brand'],sorted_info4['aggregateRating/ratingValue'],color='green')
for h in barh1:
    plt.text(h.get_width(),h.get_y()+h.get_height()/2,f'{h.get_width()}',va='center',ha='left')
plt.title('Comparing the average aggregateRating/ratingValue for each brand with horizontal barplot')
plt.xlabel('Brand',labelpad=13)
plt.ylabel('Average aggregateRating/ratingValue',labelpad=13)
plt.show()


sns.boxplot(x='brand',y='offers/price',data=data)
plt.title('Visualizing the distribution of offers/price across different brands')
plt.xlabel('Brands',labelpad=14)
plt.ylabel('Offers/price',labelpad=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


info5=data['offers/priceCurrency'].value_counts()
plt.pie(info5,labels=info5.index,autopct='%1.2f%%',startangle=55)
plt.title('The distribution of offers/priceCurrency')
plt.show()

def analyze_sentiment(text):
    blob=TextBlob(text)
    sentiment=blob.sentiment.polarity
    if sentiment<0:
        return 'Negative'
    elif sentiment==0:
        return 'Neutral'
    else:
        return 'Positive'
data['Sentiment 1']=data['features/0/description'].apply(analyze_sentiment)
data['Sentiment 2']=data['features/1/description'].apply(analyze_sentiment)
print(data['Sentiment 1'].head(15))
print(data['Sentiment 2'].head(10))
data['area']=data['width']*data['depth']
print(data['area'].head(10))
onehot=OneHotEncoder()
onehot_data=pd.DataFrame(onehot.fit_transform(data[categorical_colums]))
onehot_data_normalized=pd.concat([data.drop(columns=categorical_colums),onehot_data],axis=1)
print(onehot_data_normalized)

brand_ratings={}
for brand in data['brand'].unique():
    brand_ratings[brand]=data[data['brand']==brand]['aggregateRating/ratingValue']   
f_info,p_value=f_oneway(*brand_ratings.values())
print('Test results:\n')
print('F-statistics:\n{}'.format(f_info))
print('P-value:\n{}'.format(p_value))
alpha=0.05
if p_value<alpha:
    print('There is significant difference in rating between brands')
else:
    print('There is significant difference in rating between brands')
print('----------------------------------------------------------------')
brand_ratings2={}
for brand2 in data['brand'].unique():
    brand_ratings2[brand2]=data[data['brand']==brand2]['offers/price']
f_info2,p_value2=f_oneway(*brand_ratings2.values())
alpha2=0.06
print('Anova test results:\n')
print('F-statistics:\n{}'.format(f_info2))
print('P-value:\n{}'.format(p_value2)) 
if p_value2<alpha2:
    print('There is significant difference in offers/price between brands')
else:
    print('There is no significant difference in offers/price between brands')


x=data.drop(['offers/price', 'depth', 'width', 'features/0/description', 'features/1/description'],axis=1)
x=pd.get_dummies(x)
y=data['offers/price']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
y_predict_train=model.predict(x_train)
y_predict_test=model.predict(x_test)
trains_rmse=mean_squared_error(y_train,y_predict_train,squared=False)
test_rmse=mean_squared_error(y_test,y_predict_test,squared=False)
r2_train=r2_score(y_train,y_predict_train)
r2_test=r2_score(y_test,y_predict_test)
print('Train Rmse-{}'.format(trains_rmse))
print('Test RMSE-{}'.format(test_rmse))
print('R2 Train-{}'.format(r2_train))
print('R2 Score-{}'.format(r2_test))

