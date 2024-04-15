import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


columns = ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']
data=pd.read_csv('netflix data analysis/netflix_titles.csv',encoding='latin1',usecols=columns)
data['show_id']=[int(a.replace('s','')) for a in data['show_id']]
#print('Number of registered shows and movies-{}'.format(data.shape[0]))
def least_used_duration(columns):
    duration_count={}
    for val in columns.dropna():
        for duration in val:
            duration_count[duration]=duration_count.get(duration,0)+1
    least_used_duration1=min(duration_count,key=duration_count.get)
    return least_used_duration1
data['date_added']=pd.to_datetime(data['date_added'],errors='coerce')
data['director']=data['director'].fillna('John Worren')
data['country']=data['country'].fillna('Australia')
data['rating']=data['rating'].fillna(data['rating'].mode()[0])
data['duration']=data['duration'].fillna(least_used_duration(data['duration']))
data['date_added']=data['date_added'].fillna('2020-11-23')
data['cast']=data['cast'].fillna('Paul Hollywood, Sushar Manaying, Milan Peschel, Marlon Boess')
#print(data.isnull().sum())
#print('Is there any missing value?{}'.format(data.isnull().values.any()))
#print('Is there any duplicated value?{}'.format(data.duplicated().values.any()))
info1=data['type'].value_counts()
'''
barplot1=plt.bar(info1.index,info1.values,color='grey')
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_y(),f'{i.get_height()}',va='bottom',ha='center')
plt.title('Visualising the distribution of Type of show')
plt.xlabel('Show type',labelpad=14)
plt.ylabel('Total number of show type',labelpad=14)
plt.grid()
plt.show()
'''
'''
info2=[x for x in data['release_year'] if 1980<x<2030]
sns.histplot(info2,kde=True,color='green',bins=20)
plt.title('Analyzing the number of Tv Shows and Movies for years (1980-2025)')
plt.xlabel('Released Year',labelpad=15)
plt.ylabel('Total number of TV Shows and Movies',labelpad=15)
plt.show()
'''
info3=data['rating'].value_counts()
#print(info3)
info4=data['country'].value_counts().head(5)
#print('Top 5 countries for the description of TV shows and Movies\n{}'.format(info4))

data['Added year']=data['date_added'].dt.year
'''info5=data.groupby('Added year').size()
info5.plot(linestyle='-',marker='o',color='green')
plt.title('Investigating trends over time in terms of the year when content was added')
plt.xlabel('Year',labelpad=15)
plt.ylabel('Number of added contents',labelpad=15)
plt.grid()
plt.show()
'''
'''
plt.pie(info1,labels=info1.index,autopct='%3.2f%%')
plt.title('Generating a pie chart to visualize the distribution of type')
plt.show()
'''

all_text=' '.join(data['description'].dropna())
wordcloud1=WordCloud(width=800,height=400,background_color='white').generate(all_text)
plt.imshow(wordcloud1,interpolation='bicubic')
plt.title('Displaying most used words in description')
plt.axis('off')
plt.show()