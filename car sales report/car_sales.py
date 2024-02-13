import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('car sales report/car sales.csv')

print('number of rows-{}'.format(data.shape[0]))
print('number of columns-{}'.format(data.shape[1]))

print('is there any missing value?{}'.format(data.isnull().values.any()))
print('is there any duplicated value?{}'.format(data.duplicated().values.any()))


sns.histplot(x=data['Annual Income'],kde=True,bins=12,color='green',edgecolor='grey')
plt.title('Exploring the Distribution of Annual Income')
plt.xlabel(' ')
plt.ylabel('Annual Income')
plt.show()

sns.histplot(x=data['Price ($)'],kde=True,bins=12,color='green',edgecolor='grey')
plt.title('Exploring the Distribution of Price')
plt.xlabel(' ')
plt.ylabel('Price')
plt.show()



data=data.dropna()

data['Date']=pd.to_datetime(data['Date'],format='%d/%m/%Y',errors='coerce')
data['Dealer_No ']=data['Dealer_No '].str.replace('-','')
data['Dealer_No ']=pd.to_numeric(data['Dealer_No '],errors='coerce')
numerical_cols = data.select_dtypes(include='number')
print(numerical_cols.describe())

Q1 = data['Annual Income'].quantile(0.25)
Q3=data['Annual Income'].quantile(0.75)
iqr=Q3-Q1
lower_bound=Q1-1.5*iqr
upper_bound=Q3+1.5*iqr
info1 = data['Annual Income'].describe()
outliers=data[(data['Annual Income']<lower_bound)|(data['Annual Income']>upper_bound)][['Car_id','Customer Name','Gender']]
sns.histplot(x='Annual Income',data=data,kde=True,bins=12,color='green',edgecolor='white')
plt.title('Distribution of Annual Income without outliers')
plt.grid(True)
plt.tight_layout()
plt.show()

sns.boxplot(x=data['Annual Income'],color='green')
plt.title('Distribution of Annual Income with outliers')
plt.tight_layout()
plt.show()
print('Summary info:\n{}'.format(info1))
print('Outliers:\n{}'.format(outliers))


Q2 = data['Price ($)'].quantile(0.25)
Q4=data['Price ($)'].quantile(0.75)
iqr2=Q4-Q2
lower_bound2=Q2-1.5*iqr
upper_bound2=Q4+1.5*iqr
info2 = data['Price ($)'].describe()
outliers2=data[(data['Price ($)']<lower_bound2)|(data['Price ($)']>upper_bound2)][['Car_id','Customer Name','Gender']]
sns.histplot(x='Price ($)',data=data,kde=True,bins=12,color='green',edgecolor='white')
plt.title('Distribution of Price without outliers')
plt.grid(True)
plt.tight_layout()
plt.show()

sns.boxplot(x=data['Price ($)'],color='green')
plt.title('Distribution of Price with outliers')
plt.tight_layout()
plt.show()
print('Summary info:\n{}'.format(info2))
print('Outliers:\n{}'.format(outliers2))

info3=data['Gender'].value_counts()
barh1 = plt.barh(info3.index,info3.values,color='green')
for bar1 in barh1:
    plt.text(bar1.get_width(),bar1.get_y()+bar1.get_height()/2,f'{bar1.get_width()}',va='center',ha='left')
plt.title('Exploring the Distribution of Gender with horizontal barplot')
plt.ylabel('Gender')
plt.xlabel('Number of Male and Female')
plt.show()


info4 = sns.countplot(x='Transmission',data=data,color='blue')
for i in info4.containers:
    info4.bar_label(i)
plt.title('Exploring the distribution of transmission of cars')
plt.xlabel('Transmission type')
plt.ylabel('Number of each transmission type')
plt.grid(True)
plt.show()


info5 = sns.countplot(x='Body Style',data=data,color='blue')
for i in info5.containers:
    info5.bar_label(i)
plt.title('Exploring the distribution of body style of cars')
plt.xlabel('Body Styles')
plt.ylabel('Number of each Body style')
plt.grid(True)
plt.show()


sns.scatterplot(x='Annual Income',y='Price ($)',data=data,alpha=0.7,color='violet')
plt.title('Exploring the relationship between Annual income and Price')
plt.xticks(fontsize=10,color='blue')
plt.yticks(fontsize=10,color='blue')
plt.show()


top_10_customers = data.nlargest(10,'Annual Income')
print(top_10_customers[['Customer Name','Annual Income']])
bar2 = plt.bar(top_10_customers['Customer Name'],top_10_customers['Annual Income'],color='skyblue')
for i in bar2:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='grey')
plt.title('Identifying 10 top customers for annual salary')
plt.xlabel('Customers')
plt.ylabel('Annual Salaries')
plt.grid(True)
plt.show()


corr1 = numerical_cols.corr()
print(corr1)
sns.heatmap(corr1,annot=True)
plt.title('Visualizing the correlation matrix between numerical columns')
plt.show()


info6 = data['Dealer_Name'].value_counts()
top_5_dealers = info6.nlargest(5)
print(top_5_dealers)
top_5_dealers.plot(kind='bar',color='green')
plt.title('Exploring top 5 dealer names for the number of sold cars')
plt.xlabel('Dealer Names')
plt.ylabel('Number of sold cars')
plt.show()


info7=data['Dealer_Region'].value_counts().head(10)
sns.barplot(x=info7.index,y=info7.values,color='green')
plt.title('Exploring the distribution of Dealer regions')
plt.show()



info8=data.groupby('Dealer_Name')['Price ($)'].mean().reset_index()
sorted_info8=info8.sort_values(by='Price ($)',ascending=False)
print(sorted_info8)


data['Year']=data['Date'].dt.year
data['Month']=data['Date'].dt.month
data['Month']=data['Month'].map({
    1:'January',
    2:'February',
    3:'March',
    4:'April',
    5:'May',
    6:'June',7:'July',8:'August',9:'September',
    10:'October',11:'November',
    12:'December'
})

info9=data.groupby('Month').size().reset_index(name='Total Sales')

print(info9)
plt.plot(info9['Month'],info9['Total Sales'],marker='x',color='green',linestyle='dotted')
plt.title('Distribution of trends of sales over monthes')
plt.xlabel('Monthes')
plt.ylabel('Total Sales')
plt.show()

info10=data.groupby('Year').size().reset_index(name='Total Sales')
print(info10)


info11 = data['Model'].value_counts()
print(info11)


info12 = data.groupby('Model')['Annual Income'].mean().reset_index()
sorted_info12=info12.sort_values(by='Annual Income',ascending=False).head(10)
barplot2 = plt.bar(sorted_info12['Model'],sorted_info12['Annual Income'],color='green')
for b in barplot2:
    plt.text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height()}',va='bottom',ha='center',color='black',fontsize=7)
plt.title('Analyzing the average price for the car models')
plt.ylabel('Average Annual Income')
plt.show()


info13 = sns.countplot(x='Engine',data=data,color='grey')
for i in info13.containers:
    info13.bar_label(i)
plt.title('Exploring the distribution of engine types')
plt.show()


income = [10100,50000,100000,150000,200000,500000,800000,float('inf')]
income_Str=['10k-50k','50k-100k','100k-150k','150k-200k','200k-500k','500k-800k','800k+']
data['Income Segment']=pd.cut(data['Annual Income'],bins=income,labels=income_Str,right=False)
print(data[['Customer Name','Annual Income','Income Segment']])
    