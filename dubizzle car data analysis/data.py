import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob

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
#info3=data['body_type'].value_counts()
#print('Number of cars based on their body types:\n{}'.format(info3))
kilometers_int=[0,2000,5000,10000,20000,40000,100000,400000,1000000,2000000,float('inf')]
kilometers_str=['0-2k','2k-5k','5k-10k','10k-20k','20k-40k','40k-100k','100k-400k','400k-1M','1M-2M','2M+']
#data['driven_kilometers_category']=pd.cut(data['kilometers'],bins=kilometers_int,labels=kilometers_str,right=False)

'''
sns.countplot(x='engine_capacity_cc',data=data,color='grey',hue='warranty')
plt.title('Analyzing the number of cars based on engine capaicty and warranty')
plt.xlabel('Egine capacity',labelpad=15)
plt.ylabel('Number of cars',labelpad=15)
plt.xticks(fontsize=8)
plt.legend(loc='upper right')
plt.show()
'''
data['Price per Kilometer']=data['price']/data['kilometers']
#print(data['Price per Kilometer'])
'''
price_year=data.groupby('year')['price'].mean().reset_index()
sns.lineplot(x='year',y='price',data=price_year,color='blue',marker='o')
plt.title('Analyzing the average price over years')
plt.xlabel('Year',labelpad=14)
plt.ylabel('Average price',labelpad=14)
plt.show()
'''
'''
info4=data.groupby('area_name')['price'].mean().reset_index()
sorted_info4=info4.sort_values(by='price',ascending=False).head(15)
sns.barplot(x='area_name',y='price',data=sorted_info4,color='grey')
plt.title('Visualization the distribution of average prices of cars based on area')
plt.xlabel('Area',labelpad=14)
plt.ylabel('Average Price',labelpad=14)
plt.xticks(rotation=30,fontsize=8)
plt.show()
'''
'''
arr1=data[['kilometers','engine_capacity_cc','horsepower','no_of_cylinders']]
target_value=data['price']
arr1_encoded=pd.get_dummies(arr1)
x_train,x_test,y_train,y_test=train_test_split(arr1_encoded,target_value,random_state=42,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)

sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual price',color='green')
sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted Price',color='blue')
plt.title('Figuring out the actual price and predicted price based on features of each car')
plt.show()
'''
def sentiment_analysis(text):
    blob=TextBlob(text)
    sentiment=blob.sentiment.polarity
    if sentiment<0:
        return 'Negative'
    elif sentiment>0:
        return 'Positive'
    else:
        return 'Neutral'
data['Exterior color sentiment']=data['exterior_color'].apply(sentiment_analysis)
data['Interior color sentiment']=data['interior_color'].apply(sentiment_analysis)
#print(data['Exterior color sentiment'].head(10))
#print(data['Interior color sentiment'].head(10))
'''
countplot1=sns.countplot(x='city',data=data,hue='fuel_type')
for i in countplot1.containers:
    countplot1.bar_label(i)
plt.title('Analyzing the market share among cities')
plt.xlabel('Cities',labelpad=14)
plt.ylabel('Total number of cars',labelpad=14)
plt.legend(loc='upper right')
plt.show()
'''
competitors=['Audi','Alfa Romeo','Hyundai','Toyota']
competitors_info=data[data['brand'].isin(competitors)]
sns.barplot(x='brand',y='price',data=competitors_info,color='green')
plt.title('The comparision of prices among competeting brands')
plt.xlabel('Brands',labelpad=14)
plt.ylabel('Prices',labelpad=14)
plt.show()

sns.barplot(x='brand',y='kilometers',data=competitors_info,color='grey')
plt.title('The comparision of driven kilometers among competeting brands')
plt.xlabel('Brands',labelpad=14)
plt.ylabel('Driven kilometers',labelpad=14)
plt.show()