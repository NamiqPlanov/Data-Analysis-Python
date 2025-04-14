import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression


data=pd.read_csv('Fraud Detection Analysis/synthetic_fraud_dataset.csv')


print(data.dtypes)
'''
print('Number of rows in data - {}'.format(data.shape[0]))
print('Number of rows in data - {}'.format(data.shape[1]))
print('Is there any null value?{}'.format(data.isnull().values.any()))
print('is there any duplicated value?{}'.format(data.duplicated().values.any()))
'''

'''
numerical_columns = data.select_dtypes(include='number')
for col in numerical_columns.columns:
    print('Max value of {}-{}'.format(col,numerical_columns[col].max()))
    print('Max value of {}-{}'.format(col,numerical_columns[col].min()))
    print('Max value of {}-{}'.format(col,numerical_columns[col].median()))
    print('Max value of {}-{}'.format(col,numerical_columns[col].mean()))
    print('----------------------------------------------------------------------')
'''
'''
sns.histplot(x='Transaction_Amount',data=data,bins=12,kde=True,color='blue')
plt.title('Analyzing the amount of transactions')
plt.xlabel('Transaction Amount',labelpad=14)
plt.ylabel('')
plt.show()


sns.histplot(x='Risk_Score',data=data,kde=True,bins=12,color='green')
plt.title('Analyzing the distribution of risk score')
plt.xlabel('Scores',labelpad=14)
plt.ylabel('')
plt.show()
'''

info1 = data['User_ID'].value_counts()
top_users = info1.nlargest(10)
#print('Top Users for the number of transactions made\n{}'.format(top_users))

#Calculate the fraud rate (Fraud_Label) across different Merchant_Category values.
print(data['Merchant_Category'].unique())


info2 = data.groupby('Merchant_Category')['Fraud_Label'].mean().reset_index()
top_users_for_fraud_label = data.nlargest(10,'Fraud_Label')[['User_ID','Fraud_Label']]
#print(top_users_for_fraud_label)
'''
sns.barplot(x='Merchant_Category',y='Fraud_Label',data=info2,color='grey')
plt.title('Distribution of Fraud Label per distinct Merchant Categories')
plt.xlabel('Merchant Categories',labelpad=14)
plt.ylabel('Fraud Label',labelpad=14)
plt.tight_layout()
plt.show()
'''




