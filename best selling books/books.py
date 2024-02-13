import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report
from sklearn.model_selection import train_test_split

data=pd.read_csv('best selling books/books_info.csv')
'''
print('number of rows-{}'.format(data.shape[0]))
print('number of columns-{}'.format(data.shape[1]))
print('Is there any missing value-{}'.format(data.isnull().values.any()))
'''
data=data.dropna()
data['First_Published']=pd.to_datetime(data['First_Published'],errors='coerce')
data['First_Published']=data['First_Published'].dt.year.astype('Int64')
'''
sns.histplot(x=data['Sales_in_millions'],color='grey',kde=True,bins=11,binwidth=2,fill=False)
plt.title('Distribution of sales of books with histplot')
plt.xlabel('',labelpad=13)
plt.ylabel('Number of sales',labelpad=12)
plt.xticks(color='blue',fontsize=10)
plt.yticks(color='blue',fontsize=10)
plt.show()
'''
'''
info1=data['Language'].value_counts()
bar1=plt.bar(info1.index,info1.values,color='green')
for i in bar1:
    plt.text(i.get_x()+i.get_width()/2,i.get_height(),f'{i.get_height()}',va='bottom',ha='center',color='blue')
plt.title('Investigating the distribution of books across different languages')
plt.xlabel('Languages',labelpad=12)
plt.ylabel('Number of books',labelpad=12)
plt.tight_layout()
plt.show()
'''
'''
info2=data['Authors'].value_counts().head(8)
print(info2)
bar2=plt.barh(info2.index,info2.values,color='green')
for a in bar2:
    plt.text(a.get_width(),a.get_y()+a.get_height()/2,f'{a.get_width()}',va='center',ha='left',color='violet')
plt.title('Determining the most prolific authors')
plt.xlabel('Authors')
plt.ylabel('Number of books')
plt.show()
'''
'''
info3=data.groupby('First_Published').size().reset_index(name='Total sales')
sorted_info3=info3.sort_values(by='Total sales',ascending=False).head(3)
sns.barplot(x='First_Published',y='Total sales',data=sorted_info3,color='green')
plt.title('Finding the top years with the most number of book publishings')
plt.xlabel('Year',labelpad=14)
plt.ylabel('Number of published books',labelpad=14)
plt.show()
'''
'''
info4=data['Sales_in_millions'].sum()
print('Total sales across all books-{} milion'.format(info4))
'''
'''
info5=data.groupby('Language')['Sales_in_millions'].mean().reset_index()
sorted_info5=info5.sort_values(by='Sales_in_millions',ascending=False)
print('Computing the average sales per language:\n{}'.format(sorted_info5))
'''
'''
info6=data.groupby('Authors')['Sales_in_millions'].sum().reset_index()
sorted_info6=info6.sort_values(by='Sales_in_millions',ascending=False).head(5)
sns.barplot(x='Authors',y='Sales_in_millions',data=sorted_info6,color='green',fill=False)
plt.title('Visualizing the total sales for each author')
plt.xlabel('Author')
plt.ylabel('Sales')
plt.show()
'''
'''
info7=data.groupby('First_Published')['Sales_in_millions'].sum().reset_index()
sorted_info7=info7.sort_values(by='Sales_in_millions',ascending=False)
sorted_info7.plot(kind='line',marker='o',color='green',linestyle='dashed')
plt.title('Analyzing trends on sales over years')
plt.xlabel('Years')
plt.ylabel('Sales')
plt.show()
'''
'''
info8=data['Books'].value_counts().head(5)
sns.barplot(x=info8.index,y=info8.values,color='green')
plt.title('Displaying best selling books')
plt.xlabel('Books')
plt.ylabel('Number of sales')
plt.show()
'''
'''
info9=data['Language'].value_counts()
plt.pie(info9,labels=info9.index,autopct='%6.1f%%',startangle=35)
plt.title('Showing the distribution of books by language')
plt.show()
'''
'''
sns.scatterplot(x='First_Published',y='Sales_in_millions',data=data,alpha=0.8,color='green')
plt.title('Analyzing the relationship between year of first publishment and sales in millions')
plt.xlabel('Year of first publishment',labelpad=14)
plt.ylabel('Sales in millions',labelpad=14)
plt.show()
'''

'''
info10=data['Sales_in_millions'].describe()
print(info10)
'''

numerical_columns=data.select_dtypes(include='number')
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True)
plt.title('Analyzing the relationship between numerical columns')
plt.show()