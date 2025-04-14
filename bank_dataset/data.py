import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('bank_dataset/financial_loan.csv')


data['Issue Date'] = pd.to_datetime(data['issue_date'],format='mixed',errors='coerce')
#print(data['issue_date'].head(10))

data['Last Credit Pull Date']=pd.to_datetime(data['last_credit_pull_date'],format='mixed')
#print(data['last_credit_pull_date'].head(10))


data['Last Payment Date']=pd.to_datetime(data['last_payment_date'],format='mixed')
#print(data['last_payment_date'].head(10))


data['Next Payment Date']=pd.to_datetime(data['next_payment_date'],format='mixed')

data=data.drop(columns=['issue_date','last_credit_pull_date','next_payment_date','last_payment_date'],axis=1)


data=data.drop(['dti','total_acc','installment','int_rate'],axis=1)
print(data.dtypes)