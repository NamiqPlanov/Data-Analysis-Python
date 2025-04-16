import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
from wordcloud import WordCloud
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter




data=pd.read_csv('TwitterAnalysis/Tweets.csv')


data['negativereason'] = data['negativereason'].fillna('Unknown')
data['negativereason_confidence'] = data['negativereason_confidence'].fillna(data['negativereason_confidence'].mean())
data['airline_sentiment_gold'] = data['airline_sentiment_gold'].fillna('Unknown')
data['negativereason_gold'] = data['negativereason_gold'].fillna('Unknown')
data['tweet_location'] = data['tweet_location'].fillna('Unknown')
data['user_timezone'] = data['user_timezone'].fillna('Unknown')
data = data.drop('tweet_coord', axis=1)
data['tweet_created'] = pd.to_datetime(data['tweet_created'], format='mixed', errors='coerce')
data['tweet_created'] = data['tweet_created'].dt.tz_localize(None)

#print('Number of rows-{},columns-{}'.format(data.shape[0],data.shape[1]))
#print(data.info())
#print(data.describe())

#Count unique values in airline, airline_sentiment, negativereason
'''
print('Number of unique values in airline column-{}'.format(data['airline'].nunique()))
print('Number of unique values in airline_sentiment column - {}'.format(data['airline_sentiment'].nunique()))
print('Number of unique values in negativereason column - {}'.format(data['negativereason'].nunique()))
'''

#print(data['airline_sentiment'].value_counts())
#print(data['airline_sentiment_confidence'].mean())


#Top 10 users by tweet count (name, retweet_count)
#print(data['name'].value_counts().head(10))

#Extract hour, weekday, and month from tweet_created

data['TweetMonth'] = data['tweet_created'].dt.month
data['TweetMonth'] = data['TweetMonth'].map({
    1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
    7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'
})

data['TweetDay'] = data['tweet_created'].dt.day
data['TweetDayOfWeek'] = data['tweet_created'].dt.day_of_week
data['TweetDayOfWeek']=data['TweetDayOfWeek'].map({
    1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday'
})
data['TweetTime'] = data['tweet_created'].dt.time
data['TweetHour'] = data['tweet_created'].dt.hour

'''
info1 = data.groupby('TweetHour').size().reset_index(name='Total Tweets')
plt.plot(info1['TweetHour'],info1['Total Tweets'],marker='o',linestyle='dotted',color='green')
plt.title('Total Number of Tweets by hour')
plt.xlabel('Tweet Hours',labelpad=13)
plt.ylabel('Total Tweets',labelpad=13)
plt.show()

info2 = data.groupby('TweetDayOfWeek').size().reset_index(name='Total Tweets')
sns.barplot(x=info2['TweetDayOfWeek'],y=info2['Total Tweets'],color='green')
plt.title('Total Number of Tweets by days of week')
plt.xlabel('Tweet Days',labelpad=13)
plt.ylabel('Total Tweets',labelpad=13)
plt.show()
'''

#Bar plot of airline_sentiment count 
'''
info3 = data['airline_sentiment'].value_counts()
sns.barplot(x=info3.index,y=info3.values,color='blue')
plt.title('Distribution of Airline Sentiment')
plt.xlabel('Airline Sentiment',labelpad=13)
plt.ylabel('Number of Tweets',labelpad=13)
plt.tight_layout()
plt.show()


info4 = data['negativereason'].value_counts()
sns.barplot(x=info3.index,y=info3.values,color='grey')
plt.title('Distribution of Negative Reason')
plt.xlabel('Airline Sentiment',labelpad=13)
plt.ylabel('Number of Tweets',labelpad=13)
plt.tight_layout()
plt.show()
'''

#Group by airline and airline_sentiment — count and plot

'''countplot1 = sns.countplot(x='airline',data=data,color='grey')
for i in countplot1.containers:
    countplot1.bar_label(i)
plt.title('Number of tweets by the airline sentiment')
plt.xlabel('Airline Sentiment',labelpad=13)
plt.ylabel('Number of Tweets',labelpad=14)
plt.show()
'''

info5 = data.groupby('name')['negativereason_confidence'].mean()
#print(info5)

positive = data[data['airline_sentiment']=='positive']['airline_sentiment_confidence']
negative = data[data['airline_sentiment']=='negative']['airline_sentiment_confidence']
neutral=data[data['airline_sentiment']=='neutral']['airline_sentiment_confidence']
f_stat,p_value=f_oneway(positive,negative,neutral)
#print(f'ANOVA F-stat:{f_stat},p_value:{p_value}')


info6=data[data['tweet_location']!='Unknown']['tweet_location'].value_counts().head(10)
#print('Top 10 location for the number of tweets\n{}'.format(info6))

info7=data.groupby(['airline_sentiment','user_timezone']).size().reset_index(name='Total Tweets')
top_timezones = info7.sort_values(by='Total Tweets',ascending=False).head(10)
#print(top_timezones)

info8 = data['airline_sentiment'].value_counts()
#plt.pie(info8,labels=info8.index,autopct='%4.2f%%')
#plt.title('Pie chart of Tweets by airline sentiment')
#plt.show()

airline_sentiment = data.groupby(['airline','airline_sentiment']).size().reset_index(name='Total Tweets')
'''sns.barplot(x='airline',y='Total Tweets',data=airline_sentiment,hue='airline_sentiment')
plt.title('Sentiment distribution by Airline')
plt.xlabel('Airline',labelpad=13)
plt.ylabel('Total Tweets',labelpad=13)
plt.legend(title='Sentiment')
plt.show()'''

#WordCloud for positive and negative tweets
'''
all_text=''.join(data['text'].dropna())
wordcloud = WordCloud(width=800,height=400,background_color='white').generate(all_text)
plt.imshow(wordcloud,interpolation='bicubic')
plt.title('Wordcloud of tweets')
plt.axis('off')
plt.show()



sns.histplot(data['airline_sentiment_confidence'],kde=True,bins=12,color='grey')
plt.title('Histplot of Airline Sentiment Confidence')
plt.xlabel('Sentiment Confidence')
plt.ylabel('')
plt.show()
'''


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def process(txt):
    txt=txt.lower()
    txt=txt.translate(str.maketrans('','',string.punctuation))
    tokens=word_tokenize(txt)
    filtered=[word for word in tokens if word not in stop_words]
    return ''.join(filtered)


data['CleanedText']=data['text'].apply(process)
#print(data['CleanedText'].head(10))


def top_words(data,sentiment,n=10):
    text = ' '.join(data[data['airline_sentiment']==sentiment]['CleanedText'])
    words=text.split()
    return Counter(words).most_common(n)

top_positive = top_words(data,'positive')
top_negative=top_words(data,'negative')
top_neutral=top_words(data,'neutral')
print(top_positive)
print('-------------------------------')
print(top_negative)
print('-----------------------------------')
print(top_neutral)
print('------------------------------------')