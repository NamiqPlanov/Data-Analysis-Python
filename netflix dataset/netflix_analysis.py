import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from scipy.stats import f_oneway
from wordcloud import WordCloud

data=pd.read_csv('netflix dataset/netflix_titles.csv')
'''
print('Number of columns-{}'.format(data.shape[1]))
print('Number of rows-{}'.format(data.shape[0]))
print('Is there any missing value?{}'.format(data.isnull().values.any()))
'''
data.dropna(inplace=True)
#print('Number of rows after deleting missing values-{}'.format(data.shape[0]))
data['date_added']=pd.to_datetime(data['date_added'],errors='coerce')
'''
info1=sns.countplot(x='type',data=data,color='green',edgecolor=sns.color_palette('BrBG'),saturation=0.8)
for i in info1.containers:
    info1.bar_label(i)
plt.title('Visualising the number of each movie type in Netflix')
plt.xlabel('Movie type',labelpad=14)
plt.ylabel('Number of movie type',labelpad=14)
plt.show()
'''
'''
sns.histplot(x=data['release_year'],bins=20,kde=True,color='green',fill=False)
plt.title('Analyzing the distribution of release year with histplot')
plt.xlabel('Release year',labelpad=14)
plt.tight_layout()
plt.show()
'''

info2=data['country'].value_counts().head(10)
#print('Most common countries for the number of TV shows and movies:\n{}'.format(info2))


info3=data['rating'].value_counts()
#print('Analyzing the number of each content rating:\n{}'.format(info3))

'''
info4=data['listed_in'].value_counts().head()
barplot1=plt.bar(info4.index,info4.values,color='green')
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='grey')
plt.title('Exploring top genres')
plt.xlabel('Genres',labelpad=14)
plt.ylabel('Number of genre',labelpad=14)
plt.xticks(fontsize=5)
plt.show()
'''
'''
info5=data['release_year'].value_counts()
plt.plot(info5.index,info5.values,marker='o',linestyle='-',color='green')
plt.title('Visualising the trend of content production over years')
plt.xlabel('Years')
plt.ylabel('Trends')
plt.show()
'''
'''
sns.barplot(x=info2.index,y=info2.values,color='green',fill=False)
plt.title('Displaying top countries for movies and TV shows')
plt.xlabel('Countries',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()
'''
'''
plt.figure(figsize=(12,6))
plt.pie(info3,labels=info3.index,autopct='%1.1f%%',textprops={'fontsize': 7})
plt.title('Ploting a pie chart to visualize the distribution of rating')
plt.show()
'''
all_text=' '.join(data['listed_in'].dropna())
wordcloud1=WordCloud(width=800,height=400,background_color='white').generate(all_text)
plt.imshow(wordcloud1,interpolation='bicubic')
plt.axis('off')
plt.show()


