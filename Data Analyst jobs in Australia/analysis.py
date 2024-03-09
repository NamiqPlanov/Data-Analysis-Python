import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re


data=pd.read_csv('Data Analyst jobs in Australia/australian_data_analyst_jobs.csv')
#print('Number of rows-{}'.format(data.shape[0]))
#print('Number of columns-{}'.format(data.shape[1]))
for i in range(len(data['company_name'])):
    if i%2==0:
        if pd.isnull(data.at[i,'company_name']):
            data.at[i,'company_name']='ABC Company'
    else:
        if pd.isnull(data.at[i,'company_name']):
            data.at[i,'company_name']='UniAnalysis Company'

#print(data['company_name'][30:40])
#print('Is there any missing value?-{}'.format(data.isnull().values.any()))
#print(data.isnull().sum())
data['job_posted_date']=pd.to_datetime(data['job_posted_date'],errors='coerce')
data['job_posted_day']=data['job_posted_date'].dt.day
data['job_posted_month']=data['job_posted_date'].dt.month
data['job_posted_year']=data['job_posted_date'].dt.year
data['job_posted_month']=data['job_posted_month'].map({
    1:'January',
    2:'February'
})
'''
plt.figure(figsize=(10,7))
countplot1=sns.countplot(x='job_location/country',hue='job_salary/estimated',data=data,dodge=True)
for a in countplot1.containers:
    countplot1.bar_label(a)
plt.title('Distribution of the locations of jobs')
plt.xlabel('Job location',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.legend(title='Salary Estimated or no',loc='upper right')
plt.tight_layout()
plt.show()
'''
'''
info2=data['job_title'].value_counts().head(10)
barplot1=plt.bar(info2.index,info2.values,color='green')
for bar1 in barplot1:
    plt.text(bar1.get_x()+bar1.get_width()/2,bar1.get_height(),f'{bar1.get_height()}',va='bottom',ha='center',color='black')
plt.title('Analyzing the frequency of each job title')
plt.xlabel('Job title',labelpad=14)
plt.ylabel('Number of each job title',labelpad=14)
plt.xticks(fontsize=5)
plt.show()
'''
'''
info3=data.groupby('job_posted_month').size().reset_index(name='Total Posted Jobs')
plt.plot(info3['job_posted_month'],info3['Total Posted Jobs'],marker='o',linestyle='-',color='green')
plt.title('Exploring trends in job postings over monthes')
plt.xlabel('Monthes',labelpad=14)
plt.ylabel('Number of Posted jobs',labelpad=14)
plt.show()
'''
'''
nltk.download('stopword')
stop_words=set(stopwords.words('english'))
descriptions=' '.join(data['job_description'].dropna())
words=nltk.word_tokenize(descriptions.lower())
filtered_info=[word for word in words if words.isalnum() and word not in stop_words]
word_freq=Counter(filtered_info)
top_number=int(input('Enter number of top words:'))
top_words=word_freq.most_common(top_number)
for i,freq in top_words:
    print('{}-{}'.format(i,freq))
'''
'''
def analyze_sentiment(text):
    textblob=TextBlob(text)
    sentiment=textblob.sentiment.polarity
    if sentiment<0:
        return 'Negative'
    elif sentiment==0:
        return 'Neutral'
    else:
        return 'Positive'

data['Sentiment info']=data['job_description'].apply(analyze_sentiment)
print(data['Sentiment info'].head(10))
'''
seniority_patterns=r'\b(junior|senior|lead|manager|director)\b'
job_patterns=r'\b(engineer|developer|analyst|scinetist|desginer)\b6'
data['seniority_level']=data['job_title'].str.findall(seniority_patterns,flags=re.IGNORCASE,expand=False)
data['industry_keyword']=data['job_title'].str.findall(job_patterns,flags=re.IGNORECASE)+data['job_description'].str.findall(job_patterns,flags=re.IGNORECASE)
data['industry_keyword']=data['industry_keyword'].apply(lambda x:[keyword.lower() for keyword in x])
print('Seniority info:\n{}'.format(data['seniority_level'].value_counts()))
print('Industry keywords info:\n{}'.format(data['industry_keyword'].explode().value_counts()))