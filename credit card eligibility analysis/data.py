import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from scipy.stats import ttest_ind,f_oneway


data=pd.read_csv('credit card eligibility analysis/dataset.csv',)
data=data.drop(columns='Target',axis=1)
data['Gender']=data['Gender'].apply(lambda x:'Male' if x==1 else 'Female')
data['Own_car']=data['Own_car'].apply(lambda x:'Yes' if x==1 else 'No')
data['Own_property']=data['Own_property'].apply(lambda x:'Provided' if x==1 else 'Not provided')
data['Work_phone']=data['Work_phone'].apply(lambda x:'Provided' if x==1 else 'Not provided')
data['Phone']=data['Phone'].apply(lambda x:'Contains' if x==1 else 'Does not contain')
data['Email']=data['Email'].apply(lambda x:'Provided' if x==1 else 'Not provided')
data['Unemployed']=data['Unemployed'].apply(lambda x:'True' if x==1 else 'False')
data['Age']=data['Age'].astype('int')
data['Years_employed']=data['Years_employed'].astype('int')
#print('Is there any missing value?-{}'.format(data.isnull().values.any()))
#print('Is there any duplicated value?-{}'.format(data.duplicated().values.any()))
numerical_columns=data.select_dtypes(include='number')
print(numerical_columns.describe(include='all'))