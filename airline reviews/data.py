import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.linear_model import LinearRegression

data=pd.read_csv('airline reviews/airlines_reviews.csv')
'''
print('Number of registered reviews-{}'.format(data.shape[0]))
print('Is there any missing value?{}'.format(data.isnull().values.any()))
print(data.isnull().sum())
'''
data['Review Date']=pd.to_datetime(data['Review Date'],errors='coerce')

numerical_columns=data.select_dtypes(include='number')
#print(numerical_columns.describe())
'''
sns.histplot(x='Seat Comfort',data=data,binwidth=4,kde=True,bins=15,color='green')
plt.title('Visualisation the distribution of Seat Comfort')
plt.xlabel('Seat Comfort rating',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()

sns.histplot(x='Staff Service',data=data,binwidth=4,kde=True,bins=15,color='green')
plt.title('Visualisation the distribution of Staff Service')
plt.xlabel('Staff Service rating',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()

sns.histplot(x='Food & Beverages',data=data,binwidth=4,kde=True,bins=15,color='green')
plt.title('Visualisation the distribution of Food & Beverages')
plt.xlabel('Food & Beverages rating',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()

sns.histplot(x='Overall Rating',data=data,binwidth=4,kde=True,bins=15,color='green')
plt.title('Visualisation the distribution of Overall Rating')
plt.xlabel('Overall Rating',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()
'''
'''
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True,cmap='coolwarm')
plt.title('Analyzing correlation matrix between numerical columns with heatmap')
plt.show()
'''
'''
info1=data['Airline'].value_counts()
barplot1=plt.bar(info1.index,info1.values,color='grey')
for bar1 in barplot1:
    plt.text(bar1.get_x()+bar1.get_width()/2,bar1.get_height(),f'{bar1.get_height()}',va='bottom',ha='center')
plt.title('Visualising the frequency of each airline type')
plt.xlabel('Airline',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.xticks(fontsize=9)
plt.show()
'''
'''
info2=data['Type of Traveller'].value_counts()
barplot2=plt.barh(info2.index,info2.values,color='blue')
for barh1 in barplot2:
    plt.text(barh1.get_width(),barh1.get_y()+barh1.get_height()/2,f'{barh1.get_width()}',va='center',ha='left')
plt.title('Visualising the frequency of each type of traveller')
plt.show()
'''
'''
info3=sns.countplot(x='Class',data=data,color='green')
for i in info3.containers:
    info3.bar_label(i)
plt.title('Visualising the frequency of each class')
plt.grid(True)
plt.show()
'''

def sentiment_analysis(text):
    textblob=TextBlob(text)
    sentiment=textblob.sentiment.polarity
    if sentiment<0:
        return "Negative"
    elif sentiment==0:
        return 'Neutral'
    else:
        return 'Positive'
data['Reviews_sentiment']=data['Reviews'].apply(sentiment_analysis)
#print(data['Reviews_sentiment'].head(10))
'''
reviews=data['Reviews']

def gorupping_words(text):
    tokens=word_tokenize(text.lower())
    tokens=[word for word in tokens if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if word not in stop_words]
    lemmatizer=WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens

processed_word=[gorupping_words(review) for review in reviews]
all_tokens=[word for review_tokens in processed_word for word in review_tokens]
word_freq =Counter(all_tokens)
print('Most common words in reviews:\n{}'.format(word_freq.most_common(15)))
positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible']

positive_word_count=sum(word_freq[word] for word in positive_words)
negative_words_count=sum(word_freq[word2] for word2 in negative_words)
print('Positive words count-{}'.format(positive_word_count))
print('Negative words count-{}'.format(negative_words_count))
'''
data['Review date-month']=data['Review Date'].dt.month


data['Review date-month']=data['Review date-month'].map({
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
info4=data.groupby('Review date-month').size().reset_index(name='Overall Rating')
plt.plot(info4['Review date-month'],info4['Overall Rating'],marker='o',linestyle='-')
plt.title('Analyzing trends in overall ratings over time ')
plt.grid()
plt.show()
'''
data['Month Flown']=pd.to_datetime(data['Month Flown'],errors='coerce')
data['Month Flown-month']=data['Month Flown'].dt.month
data['Month Flown-month']=data['Month Flown-month'].map({
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
info5=data.groupby('Month Flown-month')['Overall Rating'].mean().reset_index()
sorted_info5=info5.sort_values(by='Month Flown-month')
plt.plot(sorted_info5['Month Flown-month'],sorted_info5['Overall Rating'],linestyle='-',marker='o')
plt.title('Analyzing seasonal patterns in rating over Month Flown')
plt.show()
'''

arr1=data[['Airline','Type of Traveller','Class',]]
target_data=data['Overall Rating']
arr_encoded=pd.get_dummies(arr1,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(arr_encoded,target_data,random_state=42,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
'''
sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual Rating')
sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted Rating')
plt.title('Actual vs Predicted Ratings')
plt.legend(loc='upper right')
plt.show()
'''

rmse=mean_squared_error(y_test,prediction,squared=False)
mae=mean_absolute_error(y_test,prediction)
print('Root Mean Squared Error-{}'.format(rmse))
print('Mean Absolute Error-{}'.format(mae))
