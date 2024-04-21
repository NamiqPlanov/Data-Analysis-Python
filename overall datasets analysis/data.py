import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

data=pd.read_csv('overall datasets analysis/kaggle-raw1.csv')
#print('Number of registered datasets before removing missing value-{}'.format(data.shape[0]))
data['No_of_files']=data['No_of_files'].replace('BigQuery',144)
data['No_of_files']=data['No_of_files'].astype('float')
data['No_of_files']=data['No_of_files'].fillna(5)
data['Medals']=data['Medals'].fillna('Silver')
data['Type_of_file']=data['Type_of_file'].fillna('CSV')
data['Usability']=data['Usability'].fillna(9)
data['size']=data['size'].fillna('42 MB')
data.dropna(inplace=True)
#print('Number of registered datasets after removing missing value-{}'.format(data.shape[0]))
def convert_to_num(x):
    x_str=str(x)
    value,unit=x_str[:-2],x_str[-2:].lower()
    if unit=='kb':
        return float(value)*10**-3
    elif unit=='mb':
        return float(value)
    else:
        return float(value)*10**3
    
data['size']=data['size'].apply(convert_to_num)
#print(data['size'].head(5))
#print('Is there any missing value-{}'.format(data.isnull().values.any()))
#print(data.isnull().sum())

data['Upvotes']=data['Upvotes'].astype('int')
data['Usability']=data['Usability'].astype('float')
data['Date']=pd.to_datetime(data['Date'],errors='coerce')
data['Time']=pd.to_datetime(data['Time'],format='%H:%M:%S',errors='coerce')
data['Time']=data['Time'].dt.time
numerical_cols=data.select_dtypes(include='number')
#print(numerical_cols.describe())




'''arrangement=[x for x in data['No_of_files'] if 20<x<100]
sns.histplot(arrangement,kde=True,color='green',bins=20,binwidth=3)
plt.title('Visualisation of the distribution of number of files from 20 to 100')
plt.xlabel('Number of files',labelpad=15)
plt.ylabel('Count',labelpad=15)
plt.grid()
plt.show()
'''
'''
print(numerical_cols.corr())
sns.heatmap(numerical_cols.corr(),annot=True)
plt.title('Exploring the relationship between numerical columns with hetmap')
plt.show()
'''
info1=data['Type_of_file'].value_counts().head(6)
'''sns.barplot(x=info1.index,y=info1.values,color='green')
plt.title('Analyzing Type of file using bar chart')
plt.xlabel('Type of file',labelpad=15)
plt.ylabel('Number of file',labelpad=15)
plt.tight_layout()
plt.show()
'''
info2=sns.countplot(x='Medals',data=data,color='grey')
for i in info2.containers:
    info2.bar_label(i)
plt.title('Analyzing the Number of each type of medal with barplot')
plt.xlabel('Type of Medal',labelpad=15)
plt.ylabel('Number of a medal',labelpad=15)
plt.grid()
plt.show()
