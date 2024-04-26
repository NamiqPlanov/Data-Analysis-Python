import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from textblob import TextBlob
from sklearn.linear_model import LinearRegression

data=pd.read_csv('overall datasets analysis/kaggle-raw1.csv')
print('Number of registered datasets before removing missing value-{}'.format(data.shape[0]))
data['No_of_files']=data['No_of_files'].replace('BigQuery',144)
data['No_of_files']=data['No_of_files'].astype('float')
data['No_of_files']=data['No_of_files'].fillna(5)
data['Medals']=data['Medals'].fillna('Silver')
data['Type_of_file']=data['Type_of_file'].fillna('CSV')
data['Usability']=data['Usability'].fillna(9)
data['size']=data['size'].fillna('42 MB')
data.dropna(inplace=True)
print('Number of registered datasets after removing missing value-{}'.format(data.shape[0]))
def convert_to_num(x):
    x_str=str(x)
    value,unit=x_str[:-2],x_str[-2:].lower()
    if unit=='kb':
        return float(value)*1024**-1
    elif unit=='mb':
        return float(value)
    else:
        return float(value)*1024
    
data['size']=data['size'].apply(convert_to_num)
print(data['size'].head(5))
print('Is there any missing value-{}'.format(data.isnull().values.any()))
print(data.isnull().sum())

data['Upvotes']=data['Upvotes'].astype('int')
data['Usability']=data['Usability'].astype('float')
data['Date']=pd.to_datetime(data['Date'],errors='coerce')
data['Time']=pd.to_datetime(data['Time'],format='%H:%M:%S',errors='coerce')
data['Time']=data['Time'].dt.time
numerical_cols=data.select_dtypes(include='number')
print(numerical_cols.describe())




arrangement=[x for x in data['No_of_files'] if 20<x<100]
sns.histplot(arrangement,kde=True,color='green',bins=20,binwidth=3)
plt.title('Visualisation of the distribution of number of files from 20 to 100')
plt.xlabel('Number of files',labelpad=15)
plt.ylabel('Count',labelpad=15)
plt.grid()
plt.show()


print(numerical_cols.corr())
sns.heatmap(numerical_cols.corr(),annot=True)
plt.title('Exploring the relationship between numerical columns with hetmap')
plt.show()

info1=data['Type_of_file'].value_counts().head(6)
sns.barplot(x=info1.index,y=info1.values,color='green')
plt.title('Analyzing Type of file using bar chart')
plt.xlabel('Type of file',labelpad=15)
plt.ylabel('Number of file',labelpad=15)
plt.tight_layout()
plt.show()

info2=sns.countplot(x='Medals',data=data,color='grey')
for i in info2.containers:
    info2.bar_label(i)
plt.title('Analyzing the Number of each type of medal with barplot')
plt.xlabel('Type of Medal',labelpad=15)
plt.ylabel('Number of a medal',labelpad=15)
plt.grid()
plt.show()

data['Date-Year']=data['Date'].dt.year
data['Text combined']=data['Dataset_name']+' '+data['Author_name']

tfidf=TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(data['Text combined'])
tfidf_data=pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf.get_feature_names_out())
data['Date-Month']=data['Date'].dt.month
data['Date-Month']=data['Date-Month'].map({
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

info3=data.groupby('Date-Year')['Usability'].mean()
info3.plot(linestyle='-',marker='o',color='blue')
plt.title('Analyzing trends in Usability over years')
plt.xlabel('Years',labelpad=15)
plt.ylabel('Average Usability',labelpad=15)
plt.grid()
plt.show()

info4=data.groupby('Date-Year').size()
info4.plot(linestyle='-',marker='o',color='blue')
plt.title('Analyzing trends in registered datasets over years')
plt.xlabel('Years',labelpad=15)
plt.ylabel('Nuber of registered datasets',labelpad=15)
plt.grid()
plt.show()

print(numerical_cols.corr())
sns.heatmap(numerical_cols.corr(),annot=True)
plt.title('Figuring out the relationship between numerical columns')
plt.show()

dataset_tokens=' '.join(data['Dataset_name']).split()
author_name_tokens=' '.join(data['Author_name']).split()
dataset_counter=Counter(dataset_tokens)
author_name_counter=Counter(author_name_tokens)
top_num=10
top_datasets=dataset_counter.most_common(top_num)
top_datasets_data=pd.DataFrame(top_datasets,columns=['Datasets','Frequency'])
plt.bar(top_datasets_data['Datasets'],top_datasets_data['Frequency'],color='violet')
plt.title('Figuring out top dataset word that are used')
plt.xlabel('Dataset words',labelpad=15)
plt.ylabel('Frequency',labelpad=15)
plt.grid()
plt.show()

top_authors=author_name_counter.most_common(top_num)
top_authors_data=pd.DataFrame(top_authors,columns=['Top Used Authors','Frequency'])
plt.bar(top_authors_data['Top Used Authors'],top_authors_data['Frequency'],color='orange')
plt.title('Figuring out top Author names that are used')
plt.xlabel('Author names',labelpad=15)
plt.ylabel('Frequency',labelpad=15)
plt.grid()
plt.show()

all_text=' '.join(data['Author_name'].dropna())
wordcloud1=WordCloud(width=800,height=400,background_color='white').generate(all_text)
plt.imshow(wordcloud1,interpolation='bicubic')
plt.title('Displaying most used Author Names in Dataset')
plt.show()

all_text2=' '.join(data['Dataset_name'].dropna())
wordcloud2=WordCloud(width=800,height=400,background_color='white').generate(all_text2)
plt.imshow(wordcloud2,interpolation='bicubic')
plt.title('Displaying most used Dataset Words in Dataset')
plt.show()


def analyze_text(str1):
    blob=TextBlob(str(str1))
    sentiment=blob.sentiment.polarity
    if sentiment<0:
        return 'Negative'
    elif sentiment>0:
        return 'Positive'
    else:
        return 'Neutral'
    
data['Author_name-sentiment']=data['Author_name'].apply(analyze_text)
data['Dataset_name-sentiment']=data['Dataset_name'].apply(analyze_text)
print(data['Author_name-sentiment'].head(10))
print(data['Dataset_name-sentiment'].head(10))

arr1=data[['Medals']]
target=data['Usability']
arr1_encoded=pd.get_dummies(arr1,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(arr1_encoded,target,random_state=42,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)

sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual data',color='green')
sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted data',color='blue')
plt.title('Figuring out the actual data and predicted data based on Medals of each dataset')
plt.show()


data_copied=data.copy()
data_copied.to_csv('Overall datasets analysis.csv',index=False)