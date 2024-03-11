import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy.stats import f_oneway,ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer

data=pd.read_csv('Qatar Airways Analysis/qatar_airways.csv')

print('Number of columns-{}'.format(data.shape[1]))
print('Number of rows-{}'.format(data.shape[0]))
if data.isnull().values.any()==True:
    print('\n{}'.format(data.isnull().sum()))
else:
    print('There is no missing value')


for i in range(len(data['Type Of Traveller'])):
    if i%2==0:
        if pd.isnull(data.at[i,'Type Of Traveller']):
            data.at[i,'Type Of Traveller']='Solo Leisure'
    else:
        if pd.isnull(data.at[i,'Type Of Traveller']):
            data.at[i,'Type Of Traveller']='Couple Leisure'
            
for a in range(len(data['Route'])):
    if a%2==0:
        if pd.isnull(data.at[a,'Route']):
            data.at[a,'Route']='Baku-London'
    else:
        if pd.isnull(data.at[a,'Route']):
            data.at[a,'Route']='Khanoy-Warshava'
            
data=data.drop('Recommended',axis=1)

for i in range(len(data['Aircraft'])):
    if i%2==0:
        if pd.isnull(data.at[i,'Aircraft']):
            data.at[i,'Aircraft']='Airbus-310'
    else:
        if pd.isnull(data.at[i,'Aircraft']):
            data.at[i,'Aircraft']='Boeing-442'
for i in range(len(data['Verified'])):
    if i%2==0:
        if pd.isnull(data.at[i,'Verified']):
            data.at[i,'Verified']=1
    else:
        if pd.isnull(data.at[i,'Verified']):
            data.at[i,'Verified']=0




data['Date Published'] = pd.to_datetime(data['Date Published'], errors='coerce')
data.loc[1935:2030, 'Date Flown'] = pd.to_datetime('23/03/2024', dayfirst=True)
data.loc[2031:2067, 'Date Flown'] = pd.to_datetime('27/10/2023', dayfirst=True)
data.loc[2068:2170, 'Date Flown'] = pd.to_datetime('12/04/2023', dayfirst=True)
data.loc[2171:2270, 'Date Flown'] = pd.to_datetime('05/12/2023', dayfirst=True)
data.loc[2172:2245, 'Date Flown'] = pd.to_datetime('12/04/2019', dayfirst=True)
data.loc[2245:2368, 'Date Flown'] = pd.to_datetime('06/02/2020', dayfirst=True)
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')
data['Date Flown'] = pd.to_datetime(data['Date Flown'],format='%d/%m/%Y',errors='coerce')
data=data.dropna()
data['Rating']=data['Rating'].astype('int')
data['Max Rating']=data['Max Rating'].astype('int')
numerical_columns=data.select_dtypes(include='number')
print(numerical_columns.describe())

sns.histplot(x='Rating',data=data,binwidth=5,bins=20,kde=True,color='green',shrink=.7)
plt.title('Visualizing the distribution of ratings')
plt.xlabel('Ratings',labelpad=15)
plt.ylabel('',labelpad=16)
plt.grid(True)
plt.show()





data['Date Month']=data['Date'].dt.month
data['Date Year']=data['Date'].dt.year
data['Date Day']=data['Date'].dt.day
data['Date Day']=data['Date Day'].map({
    1:'Monday',
    2:'Tuesday',
    3:'Wednesday',
    4:'Thursday',
    5:'Friday',
    6:'Saturday',
    7:'Sunday'
})
data['Date Month']=data['Date Month'].map({
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

info1=data.groupby('Date Year')['Rating'].mean().reset_index()
sns.barplot(x='Date Year',y='Rating',data=info1,color='grey')
plt.title('Analyzing the distribution of ratings over time')
plt.xlabel('Date',labelpad=16)
plt.ylabel('Rating',labelpad=16)
plt.grid()
plt.show()

info2=data.groupby('Country')['Rating'].mean().reset_index()
sorted_info2=info2.sort_values(by='Rating',ascending=False).head(15)
print(sorted_info2)

sns.barplot(x='Country',y='Rating',data=sorted_info2,color='green')
plt.title('Exploring the distribution of ratings by country')
plt.xlabel('Country',labelpad=16)
plt.ylabel('Rating',labelpad=16)
plt.xticks(fontsize=9)
plt.grid()
plt.show()


sns.scatterplot(x='Rating',y='Max Rating',data=data,color='blue',alpha=0.5)
plt.title('Investigating the relationship between Rating and Max Rating')
plt.show()


def sentiment_analysis(text):
    textblob=TextBlob(text)
    sentiment=textblob.sentiment.polarity
    if sentiment<0:
        return 'Negative'
    elif sentiment==0:
        return 'Zero'
    else:
        return 'Positive'
data['Review sentiment']=data['Review Body'].apply(sentiment_analysis)
print(data['Review sentiment'].head(15))




data['Date Published Month']=data['Date Published'].dt.month
data['Date Published Year']=data['Date Published'].dt.year
data['Date Published Day']=data['Date Published'].dt.day
data['Date Published Day']=data['Date Published Day'].map({
    1:'Monday',
    2:'Tuesday',
    3:'Wednesday',
    4:'Thursday',
    5:'Friday',
    6:'Saturday',
    7:'Sunday'
})
data['Date Published Month']=data['Date Published Month'].map({
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



data['Date Flown Month']=data['Date Flown'].dt.month
data['Date Flown Year']=data['Date Flown'].dt.year
data['Date Flown Day']=data['Date Flown'].dt.day
data['Date Flown Day']=data['Date Flown Day'].map({
    1:'Monday',
    2:'Tuesday',
    3:'Wednesday',
    4:'Thursday',
    5:'Friday',
    6:'Saturday',
    7:'Sunday'
})
data['Date Flown Month']=data['Date Flown Month'].map({
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

data['Verified or no']=data['Verified'].astype('bool')

categorical_columns=['Type Of Traveller', 'Seat Type', 'Route', 'Aircraft']
onehot=OneHotEncoder()
onehot_encoder=pd.DataFrame(onehot.fit_transform(data[categorical_columns]))
onehot_normalized=pd.concat([data.drop(columns=categorical_columns),onehot_encoder],axis=1)
print(onehot_normalized.head(5))

info2=data['Type Of Traveller'].value_counts()
print(info2)

barplot1=plt.bar(info2.index,info2.values,color='blue')
for bar1 in barplot1:
    plt.text(bar1.get_x()+bar1.get_width()/2,bar1.get_height(),f'{bar1.get_height()}',va='bottom',ha='center')
plt.title('Analyzing the distribution of Traveller type')
plt.xlabel('Traveller type',labelpad=16)
plt.ylabel('Number of each type',labelpad=16)
plt.show()


info3=data['Seat Type'].value_counts()

barplot2=plt.bar(info3.index,info3.values,color='blue')
for bar2 in barplot2:
    plt.text(bar2.get_x()+bar2.get_width()/2,bar2.get_height(),f'{bar2.get_height()}',va='bottom',ha='center')
plt.title('Analyzing the distribution of Seat type')
plt.xlabel('Seat type',labelpad=16)
plt.ylabel('Number of each seat type',labelpad=16)
plt.show()


all_text=' '.join(data['Review Body'].dropna())
wordcloud=WordCloud(width=800,height=500,background_color='white').generate(all_text)
plt.imshow(wordcloud,interpolation='bicubic')
plt.axis('off')
plt.show()

arr_category=data[['Author','Country','Aircraft']]
target=data['Rating']
arr_encoded=pd.get_dummies(arr_category,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(arr_encoded,target,random_state=42,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
mse=mean_squared_error(y_test,prediction)
print('Mean Squared Error for rating-{}'.format(mse))
sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual Rating')
sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted Rating')
plt.title('Actual vs Predicted Ratings')
plt.legend(loc='upper right')
plt.show()

data_copy=data.copy()
data_copy.to_csv('Qatar Airways modified.csv',index=False)
