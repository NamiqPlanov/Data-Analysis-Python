import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy.stats import f_oneway,ttest_ind
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv('study performance analysis/study_performance.csv')
#print('Number of rows-{}'.format(data.shape[0]))
#print('Number of columns-{}'.format(data.shape[1]))
#print('Is there any missing value-{}'.format(data.isnull().values.any()))
'''
print('Numerical analysis for math score:\n{}'.format(data['math_score'].describe()))
print('Numerical analysis for reading score:\n{}'.format(data['reading_score'].describe()))
print('Numerical analysis for writing score:\n{}'.format(data['writing_score'].describe()))
'''
'''
sns.histplot(x='math_score',data=data,kde=True,color='green',bins=15,binwidth=5)
plt.title('Visualisation the distribution of math score')
plt.xlabel('Math score',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()
'''
'''
sns.histplot(x='reading_score',data=data,kde=True,color='grey',element='step',bins=25,shrink=.5)
plt.title('Visualisation the distribution of reading score')
plt.xlabel('Reading score',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()

sns.histplot(x='writing_score',data=data,kde=True,color='violet',element='poly',bins=20,shrink=.5)
plt.title('Visualisation the distribution of writing score')
plt.xlabel('Writing score',labelpad=14)
plt.ylabel('Count',labelpad=14)
plt.show()
'''
'''
info1=data['gender'].value_counts()
barplot1=plt.bar(info1.index,info1.values,color='blue',fill=False)
for bar1 in barplot1:
    plt.text(bar1.get_x()+bar1.get_width()/2,bar1.get_height(),f'{bar1.get_height()}',va='bottom',ha='center')
plt.title('Analyzing distribution of gender')
plt.xlabel('Gender',labelpad=14)
plt.ylabel('Number of each gender',labelpad=14)
plt.tight_layout()
plt.show()

info2=data['race_ethnicity'].value_counts()
barplot2=plt.bar(info2.index,info2.values,color='blue',fill=True)
for bar2 in barplot2:
    plt.text(bar2.get_x()+bar2.get_width()/2,bar2.get_height(),f'{bar2.get_height()}',va='bottom',ha='center')
plt.title('Analyzing distribution of race ethnicity')
plt.xlabel('Race Ethnicity',labelpad=14)
plt.ylabel('',labelpad=14)
plt.tight_layout()
plt.show()


info3=data['parental_level_of_education'].value_counts()
barplot3=plt.barh(info3.index,info3.values,color='yellow',fill=True)
for bar3 in barplot3:
    plt.text(bar3.get_width(),bar3.get_y()+bar3.get_height()/2,f'{bar3.get_width()}',va='center',ha='left')
plt.title('Analyzing distribution of level of education')
plt.xlabel('Level of education',labelpad=14)
plt.ylabel('Number of each education level',labelpad=14)
plt.show()
'''
'''
sns.scatterplot(x='math_score',y='reading_score',data=data,color='green',alpha=0.9)
plt.title('Relationship between Math and Reading scores')
plt.show()
'''
numerical_columns=data.select_dtypes(include='number')
'''
print(numerical_columns.corr())
sns.heatmap(numerical_columns.corr(),annot=True,cmap='coolwarm')
plt.title('Analyzing the correlation matrix between numerical columns with heatmap')
plt.show()
'''

data['Total score']=data['math_score']+data['reading_score']+data['writing_score']

'''
info4=data.groupby('gender')['Total score'].mean().reset_index()
sns.barplot(x='gender',y='Total score',data=info4,color='green')
plt.title('Analyzing the average scores for each gender')
plt.xlabel('Gender',labelpad=16)
plt.ylabel('Average total score',labelpad=16)
plt.xticks(rotation=45)
plt.show()

info5=data.groupby('parental_level_of_education')['Total score'].mean().reset_index()
sns.barplot(x='parental_level_of_education',y='Total score',data=info5,color='green')
plt.title('Analyzing the average scores for each  level of education')
plt.xlabel('Level of education',labelpad=16)
plt.ylabel('Average total score',labelpad=16)
plt.xticks(rotation=45)
plt.show()

info6=data.groupby('lunch')['Total score'].mean().reset_index()
sns.barplot(x='lunch',y='Total score',data=info6,color='green')
plt.title('Analyzing the average scores for each lunch')
plt.xlabel('Lunch',labelpad=16)
plt.ylabel('Average total score',labelpad=16)
plt.xticks(rotation=45,fontsize=9)
plt.show()
'''
'''
education_level_ratings={}
for i in data['parental_level_of_education'].unique():
    education_level_ratings[i]=data[data['parental_level_of_education']==i]['Total score']
f_info,p_value=f_oneway(*education_level_ratings.values())
alpha=0.07
print('Anova test results')
print('F-statistics-{}'.format(f_info))
print('P-value-{}'.format(p_value))
if p_value<alpha:
    print('There is significant difference in total score between education levels')
else:
    print('There is no significant difference in total score between education levels')


completed_test=data[data['test_preparation_course']=='completed']['Total score']
not_completed_test=data[data['test_preparation_course']=='none']['Total score']
t_stat,p_value2=ttest_ind(completed_test,not_completed_test)
alpha2=0.05
if p_value2<alpha2:
    print('There is significant difference in total score between complted test course and not completed test courses')
else:
    print('There is no significant difference in total score between complted test course and not completed test courses')
'''
'''
ethnicity={}
for a in data['race_ethnicity'].unique():
    ethnicity[a]=data[data['race_ethnicity']==a]['Total score']
f_info2,p_value3=f_oneway(*ethnicity.values())
alpha3=0.06
if p_value3<alpha3:
    print('There is significant difference in total score between ethnicity groups')
else:
    print('There is no significant difference in total score between ethnicity groups')
    '''
'''
def prediction_data(info):
    arr2=data[['gender','lunch','race_ethnicity','parental_level_of_education']]
    target=data[info+'_score']
    feature_encoded=pd.get_dummies(arr2,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(feature_encoded,target,random_state=42,test_size=0.3)
    model=LinearRegression()
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    mse=mean_squared_error(y_test,prediction)
    print('Mean Squared Error for {}-{}'.format(info,mse))
    sns.lineplot(x=range(len(y_test)),y=y_test.values,label='Actual score')
    sns.lineplot(x=range(len(prediction)),y=prediction,label='Predicted score')
    plt.title('Actual and Predicted scores')
    plt.legend()
    plt.show()


prediction_data('math')
prediction_data('writing')
'''



def cross_validation(subject):
    features = data[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
    target = data[subject+'_score']
    features_encoded = pd.get_dummies(features, drop_first=True)
    model = LinearRegression()
    cv_scores = cross_val_score(model, features_encoded, target, cv=5, scoring='neg_mean_squared_error')
    mse = -cv_scores.mean()
    print("Mean Squared Error for {} Scores Prediction (Cross-Validation): {:.2f}".format(subject.capitalize(), mse))
    
cross_validation('math')
