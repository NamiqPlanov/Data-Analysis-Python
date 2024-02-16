import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from textblob import TextBlob
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report
from scipy.stats import f_oneway
from statsmodels.tsa.arima.model import ARIMA 

data=pd.read_csv('urban population(1950-2050)/dataset.csv')
#print('number of rows-{}'.format(data.shape[0]))
data=data.dropna(inplace=True)
#data['Absolute value in thousands']=data['Absolute value in thousands'].replace('NOT APPLICABLE',pd.NA)
data['Absolute value in thousands Missing value']=data['Absolute value in thousands Missing value'].replace('NOT APPLICABLE',pd.NA)
data['Urban population as percentage of total population']=data['Urban population as percentage of total population'].replace('NOT APPLICABLE',pd.NA)
data['Urban population as percentage of total population Missing value']=data['Urban population as percentage of total population Missing value'].replace('NOT APPLICABLE',pd.NA)
data=data.dropna(inplace=True)
#data['Absolute value in thousands']=data['Absolute value in thousands'].astype('float')
data['Absolute value in thousands Missing value']=data['Absolute value in thousands Missing value'].astype('int64')
data['Urban population as percentage of total population']=pd.to_numeric(data['Urban population as percentage of total population'],errors='coerce')


'''
print('Analyzing the statistics of Year:')
print(data['Year'].describe())
print('-----------------------------------')
print('Analyzing the statistics of Economy:')
print(data['Economy'].describe())
'''
'''
info1=data['Economy Label'].value_counts()
print(info1)
'''

numerical_columns=data.select_dtypes(include='number')
'''
for i in numerical_columns:
    sns.histplot(data=data,x=i,bins=13,kde=True,color='green')
    plt.title('Analyzing the distribution of {}'.format(i))
    plt.xlabel(i,labelpad=14)
    plt.ylabel('Count',labelpad=14)
    plt.show()
'''
'''
info2=data['Economy Label'].value_counts().head(10)
barplot1=plt.bar(info2.index,info2.values,color='green',fill=False)
for i in barplot1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='grey')
plt.title('Visualizing the distribution of Economy label')
plt.xlabel('Country')
plt.ylabel('Economy')
plt.xticks(fontsize=8)
plt.show()
'''
'''
data['Year']=pd.to_datetime(data['Year'],errors='coerce')
plt.plot(data['Year'],data['Economy'],marker='o',color='green',linestyle='dotted')
plt.title('Analyzing time series changes of ecenomy over year')
plt.show()
'''
'''
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True)
plt.title('Analyzing the relationship between numerical columns')
plt.show()
'''
'''
info3=data.groupby('Year')['Urban population as percentage of total population'].mean().reset_index()
sorted_info3=info3.sort_values(by='Urban population as percentage of total population',ascending=False).head(5)
plt.plot(sorted_info3['Year'],sorted_info3['Urban population as percentage of total population'],marker='o',linestyle='dashed')
plt.show()
'''
'''
sns.scatterplot(x='Economy',y='Urban population as percentage of total population',data=data,alpha=0.8,color='green')
plt.title('Investigating the relationship between Economy and urban population as percentage')
plt.xlabel('Economy')
plt.ylabel('Urban population as percentage')
plt.show()
'''
'''
year_1=int(input('Input your first year:'))
year2=int(input('Input your second year:'))
specific_data=(data['Year']>year_1)&(data['Year']<year2)
filtered_data=data[specific_data]
print(filtered_data)
'''
'''
data['Economy Label']=data['Economy Label'].astype('str')
label=['Algeria']
data2=data['Economy Label'].isin(label)
filtered_data2=data[data2]
print(filtered_data2)
label_count=data['Economy Label'].value_counts()[label]
print(label_count)
'''

'''
unique_labels=data['Economy Label'].unique()
new_list=[]
for i in unique_labels:
    new_list.append(data[data['Economy Label']==i]['Urban population as percentage of total population'])

f_statistics,p_value=f_oneway(*new_list)
alpha=0.07
if p_value<alpha:
    print('There is significant difference between Economy labels and Urban population as percentage')
else:
    print('There is no significant difference between Economy labels and Urban population as percentage')
'''
'''
corr2=data['Economy'].corr(data['Urban population as percentage of total population'])
print('The correlation between Economy and Urban population as percentage:{}'.format(corr2))
'''
data.set_index(data['Year'],inplace=True)
training_size=int(len(data)*0.8)
training_data,test_data=data.iloc[:training_size],data.iloc[training_size:]
model=ARIMA(training_data,order=(5,1,0))
arima_model=model.fit()
forecast=arima_model.forecast(steps=len(test_data))
mse=mean_squared_error(test_data,forecast)
rmse=np.sqrt(mse)
print('RMSE:{}'.format(rmse))
plt.plot(training_data,label='Training data')
plt.plot(test_data.index,test_data,label='Actual data')
plt.plot(test_data.index,forecast,label='Predicted data')
plt.title('Urban Population Percentage Forecast using ARIMA')
plt.xlabel('Year')
plt.ylabel('Urban Population Percentage')
plt.legend()
plt.show()