import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


data = pd.read_csv('bankTransactions/banktransactions.csv')
print(data.head(10))

 
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'],format='mixed')
data['TransactionDay'] = data['TransactionDate'].dt.day_of_week
data['TransactionDay'] = data['TransactionDay'].map({
    0:'Monday',1:'Tuesday',
    2:'Wednesday',3:'Thursday',
    4:'Friday',5:'Saturday',6:'Sunday'
})

data['TransactionMonth'] = data['TransactionDate'].dt.month
data['TransactionMonth'] = data['TransactionMonth'].map({
    1:'January',2:'February',3:'March',4:'April',5:'May',
    6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',
    12:'December'
})

data['PreviousTransactionDate'] = pd.to_datetime(data['PreviousTransactionDate'],format='mixed')
data['PreviousTransactionDate'] = data['PreviousTransactionDate'].dt.day_of_week
data['PreviousTransactionDate'] = data['PreviousTransactionDate'].map({
    0:'Monday',1:'Tuesday',
    2:'Wednesday',3:'Thursday',
    4:'Friday',5:'Saturday',6:'Sunday'
})



print('Is there any duplicated row? {}'.format(data.duplicated().values.any()))


print('Is there any duplicated row? {}'.format(data.isnull().values.any()))



print('The mean value of TransactionAmount - {}'.format(data['TransactionAmount'].mean()))
print('The median of TransactionAmount - {}'.format(data['TransactionAmount'].median()))
print('The Standard Deviation of TransactionAmount - {}'.format(data['TransactionAmount'].std()))
print('-----------------------------------------------')

print('The mean value of CustomerAge - {}'.format(data['CustomerAge'].mean()))
print('The median of CustomerAge - {}'.format(data['CustomerAge'].median()))
print('The Standard Deviation of CustomerAge - {}'.format(data['CustomerAge'].std()))
print('-----------------------------------------------')

print('The mean value of AccountBalance - {}'.format(data['AccountBalance'].mean()))
print('The median of AccountBalance - {}'.format(data['AccountBalance'].median()))
print('The Standard Deviation of AccountBalance - {}'.format(data['AccountBalance'].std()))



print(data.nunique())



sns.histplot(data['TransactionAmount'],kde=True,bins=11,color='green')
plt.title('Histplot of the distribution of The Amount of Transaction')
plt.tight_layout()
plt.show()

sns.histplot(data['CustomerAge'],kde=True,bins=11,color='blue')
plt.title('Histplot of the distribution of The Age of Customers')
plt.xlabel('Ages of Customers')
plt.tight_layout()
plt.show()

sns.histplot(data['TransactionDuration'],kde=True,bins=11,color='blue')
plt.title('Histplot of the distribution of The Duration of Transactions')
plt.xlabel('Durations of Transactions')
plt.tight_layout()
plt.show()


numerical_cols = data.select_dtypes(include='number')

sns.jointplot(x='AccountBalance',y='TransactionAmount',data=data,color='orange',kind='hex')
plt.title('Joint Plot of 2 numerical columns')
plt.tight_layout()
plt.show()


dailyTransactions = data.groupby('TransactionDay').size().reset_index(name='TotalTransactions')
sns.barplot(x='TransactionDay',y='TotalTransactions',data=dailyTransactions,color='green')
plt.title('Transactions per day')
plt.xlabel('Total number of Transactions')
plt.tight_layout()
plt.show()


countplot1 = sns.countplot(x='TransactionMonth',data=data,color='yellow')
for i in countplot1.containers:
    countplot1.bar_label(i)
plt.title('Transactions per month')
plt.xlabel('Months')
plt.ylabel('Number of transactions')
plt.show()




topLocations = data.groupby(['Location','TransactionType']).size().reset_index(name='TotalTransactions')
sortedlocations = topLocations.nlargest(10,'TotalTransactions')
print(sortedlocations)



sns.barplot(x='Location',y='TotalTransactions',hue='TransactionType',data=sortedlocations,color='blue')
plt.title('Top Locations for the Money Transaction')
plt.legend(loc='upper right')
plt.show()
print(sortedlocations)



avgtransaction = data.groupby('TransactionID')['TransactionAmount'].mean().reset_index().head(15)
print(avgtransaction)



data['TransactionHour'] = data['TransactionDate'].dt.hour
transactionPerHour = data.groupby('TransactionHour').size().reset_index(name='TotalTransactions').head(3)
sns.barplot(x='TransactionHour',y='TotalTransactions',data=transactionPerHour,color='blue')
plt.title('Analyzing  3 peakest transaction hours')
plt.xlabel('Hours')
plt.ylabel('Number of Transactions')
plt.show()


typesOfTransaction = data['TransactionType'].value_counts()
plt.pie(typesOfTransaction,labels=typesOfTransaction.index,autopct='%2.1f%%')
plt.title('Types of Transaction')
plt.show()


channelInfo = data['Channel'].value_counts()
print(channelInfo)



specificCols= data[['CustomerAge','LoginAttempts','AccountBalance']]



maxAttempts = 3
print('Number of Individuals who attempted more than 3 times-{}'.format(data[data['LoginAttempts']>3]['LoginAttempts'].sum()))
print('Details of some of them\n{}'.format(data[data['LoginAttempts']>3][['TransactionID','AccountID','TransactionType','LoginAttempts']].head(10)))



minDuration = 150
maxDuration = 260
print('Number of People who spent less time than required-{}'.format(len(data[data['TransactionDuration']<minDuration])))
print('Number of People who spent more time than required-{}'.format(len(data[data['TransactionDuration']<maxDuration])))



customersWithUniqueIPs = data.groupby('AccountID')['IP Address'].nunique().reset_index()
customersWithUniqueIPs.columns = ['AccountID', 'DistinctIPCount']
customersWithMultipleIPs = customersWithUniqueIPs[customersWithUniqueIPs['DistinctIPCount']>1]['AccountID'].head(15)
print(customersWithMultipleIPs)




ages = [18,35,50,65,float('inf')]
age_group = ['18-35','35-50','50-65','65+']
data['AgeGroup'] = pd.cut(data['CustomerAge'],bins=ages,labels=age_group,right=False)

transactions =[0,100,200,400,800,1600,float('inf')]
transactionGroup = ['0 transaction','very little transaction','small transaction',
'medium transaction','large transaction','extremely large transaction']
data['TransactionGroup'] = pd.cut(data['TransactionAmount'],bins=transactions,labels=transactionGroup,right=False)
print(data['TransactionGroup'].head(10))


topMerchants = data.nlargest(5,'TransactionAmount')[['TransactionID','AccountID','TransactionAmount']]
print(topMerchants)



AvgTransaction = data.groupby('TransactionDay')['TransactionAmount'].mean().reset_index()
print(AvgTransaction)




data['ChangeInBalance'] = data.groupby('AccountID')['TransactionAmount'].diff()
print(data[data['ChangeInBalance']>0][['AccountID','TransactionDate','TransactionAmount', 'AccountBalance', 'ChangeInBalance']].head(15))


TopDevices = data['DeviceID'].value_counts().head(5).index
filteredData = data[data['DeviceID'].isin(TopDevices)]

sns.countplot(x='DeviceID',hue='TransactionType',data=filteredData,color='green')
plt.title('Top 5 devices which were used for Transactions')
plt.xlabel('Devices')
plt.ylabel('Number of Transactions')
plt.legend(loc='upper right')
plt.show()


datacopied=data.copy()
datacopied.to_csv('BankTransactions.csv',index=False)