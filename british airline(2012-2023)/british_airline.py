import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('british airline(2012-2023)/british_airline.csv')

print('number of rows-{}'.format(data.shape[0]))
print('number of columns-{}'.format(data.shape[1]))
print('is there any missing value?{}'.format(data.isnull().values.any()))
print(data.isnull().sum())
print('is there any duplicates?{}'.format(data.duplicated().values.any()))


data=data.dropna()
data['date']=pd.to_datetime(data['date'],format='%d-%m-%Y',errors='coerce')
data['date_flown']=pd.to_datetime(data['date_flown'],errors='coerce')
numerical_columns = data.select_dtypes(include='number')
print(numerical_columns.describe())

sns.histplot(x=data['rating'],kde=True,color='green',bins=11,edgecolor='green')
plt.title('Exploring the distribution of ratings with histplot')
plt.xticks(color='green',fontsize=10)
plt.yticks(color='green',fontsize=10)
plt.grid(True)
plt.show()


sns.boxplot(x=data['seat_comfort'],color='green')
plt.title('Exploring the distribution of seat comfort with boxplot')
plt.xticks(color='green',fontsize=10)
plt.yticks(color='green',fontsize=10)
plt.grid(True)
plt.show()

sns.histplot(x=data['cabin_staff_service'],kde=True,color='blue',bins=11,edgecolor='white')
plt.title('Exploring the distribution of cabin staff service with histplot')
plt.xlabel('Cabin staff service rating',labelpad=12)
plt.ylabel('Count',labelpad=12)
plt.xticks(color='grey',fontsize=10)
plt.yticks(color='grey',fontsize=10)
plt.grid(True)
plt.show()


sns.histplot(x=data['food_beverages'],kde=True,color='blue',bins=11,edgecolor='white')
plt.title('Exploring the distribution of food and beverages with histplot')
plt.xlabel('Food beverages rating',labelpad=12)
plt.ylabel('Count',labelpad=12)
plt.xticks(color='grey',fontsize=10)
plt.yticks(color='grey',fontsize=10)
plt.grid(True)
plt.show()


sns.histplot(x=data['ground_service'],kde=True,color='blue',bins=11,edgecolor='white')
plt.title('Exploring the distribution of ground service rating with histplot')
plt.xlabel('ground service rating',labelpad=12)
plt.ylabel('Count',labelpad=12)
plt.xticks(color='green',fontsize=10)
plt.yticks(color='green',fontsize=10)
plt.grid(True)
plt.show()

info1 = data['recommended'].value_counts()
plt.pie(info1,labels=info1.index,startangle=45,autopct='%4.2f%%')
plt.title('Exploring the classification of recommended column of the dataset')
plt.show()


info2 = data['trip_verified'].value_counts()
plt.pie(info2,labels=info2.index,startangle=45,autopct='%5.2f%%')
plt.title('Exploring the classification of the trip which is verified or not')
plt.show()


plt.figure(figsize=(9,7))
sns.scatterplot(x='rating',y='seat_comfort',data=data,alpha=0.8,color='blue')
plt.title('Exploring the relationship between rating and seat comfort columns of the dataset')
plt.xlabel('Rating',labelpad=12)
plt.ylabel('Seat Comfort',labelpad=12)
plt.xticks(color='green',fontsize=10)
plt.yticks(color='green',fontsize=10)
plt.show()


all_text=' '.join(data['content'].dropna())
wordcloud = WordCloud(width=800,height=400,background_color='white').generate(all_text)
plt.imshow(wordcloud,interpolation='bicubic')
plt.title('Wordcloud of content')
plt.axis('off')
plt.show()


arr1 = data[['rating','seat_comfort','cabin_staff_service','food_beverages','ground_service','value_for_money','entertainment']]
corr1 = arr1.corr()
sns.heatmap(arr1.corr(),annot=True)
plt.title('Exploring the relationship between numerical columns with correlation matrix')
plt.show()


info3=data.groupby('traveller_type')['rating'].mean().reset_index()
barplot1 = plt.bar(info3['traveller_type'],info3['rating'],color='green')
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='grey')
plt.title('Exploring the distribution of average rating for travller type')
plt.xlabel('Traveller type',labelpad=13)
plt.ylabel('Average Rating',labelpad=13)
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()


info4=data.groupby('seat_type')['rating'].mean().reset_index()
barplot2 = plt.bar(info4['seat_type'],info4['rating'],color='green')
for i in barplot2:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='grey')
plt.title('Exploring the distribution of average rating for seat type')
plt.xlabel('Seat type',labelpad=13)
plt.ylabel('Average Rating',labelpad=13)
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()


info5=data.groupby('place')['rating'].mean().reset_index()
sorted_info5 = info5.sort_values(by='rating',ascending=False).head(6)
barplot3 = plt.bar(sorted_info5['place'],sorted_info5['rating'],color='green')
for i in barplot3:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='grey')
plt.title('Exploring the distribution of average rating for seat type')
plt.xlabel('Place',labelpad=13)
plt.ylabel('Average Rating',labelpad=13)
plt.xticks(color='blue',fontsize=11)
plt.yticks(color='blue',fontsize=11)
plt.show()


data['Year flown']=data['date_flown'].dt.year
data['Month flown']=data['date_flown'].dt.month
seasonal_flights=data.groupby(['Year flown','Month flown']).size()
seasonal_flights.plot(linestyle='dashed',marker='o',color='green')
plt.title('Analyzing seasonal trends in flights')
plt.xlabel('Year and month',labelpad=13)
plt.ylabel('Number of flights',labelpad=13)
plt.show()


def analyze_sentiment(text):
    blob = TextBlob(str(text))
    sentiment_score=blob.sentiment.polarity
    if sentiment_score<0:
        return 'Negative'
    elif sentiment_score>0:
        return 'Positive'
    else:
        return 'Neutral'

data['sentiment']=data['content'].apply(analyze_sentiment)
sentiment_count=data['sentiment'].value_counts()
print(sentiment_count)


def analyze_rating(text):
    blob=TextBlob(str(text))
    rating_score=blob.sentiment.polarity
    if rating_score<0:
        return 'Negative'
    elif rating_score>0:
        return 'Positive'
    else:
        return 'Neutral'
data['sentiment']=data['content'].apply(analyze_rating)

sentiment_info=data.groupby(['rating','sentiment']).size().unstack(fill_value=0)
normalized_info = sentiment_info.div(sentiment_info.sum(axis=1),axis=0)*100
print("Sentiment Distribution Across Different Ratings:")
print(normalized_info)

columns = ['seat_comfort', 'cabin_staff_service', 'food_beverages', 'ground_service', 'value_for_money', 'entertainment']

target='rating'
x_train,x_test,y_train,y_test=train_test_split(data[columns],data[target],test_size=0.3,random_state=43)
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=43)
rf_classifier.fit(x_train,y_train)
feature_importance=  rf_classifier.feature_importances_
plt.bar(columns,feature_importance)
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.show()  


target2 = 'recommended'
x_train2,x_test2,y_trains2,y_test2=train_test_split(data[columns],data[target2],test_size=0.2,random_state=42)
rf_classifier2 = RandomForestClassifier(n_estimators=100,random_state=42)
rf_classifier2.fit(x_train2,y_trains2)
feature_importance2 = rf_classifier2.feature_importances_
plt.bar(columns,feature_importance2)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')  
plt.show()

