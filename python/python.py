import pandas as pd
from pandas import Series,DataFrame
import numpy as np


arr=pd.Series([2,3,4,5,6])
#print(arr)
#print(arr.values)

arr2=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
#print(arr2)
#print(arr2.index)
#print(arr2[arr2>3])
#print(arr2*3)
#
#print('f' in arr2)
'''
arr3={"baku":3000,'Shamakhi':4000,'Quba':5000}
arr4=pd.Series(arr3)
print(arr4)
arr_true_false=pd.isnull(arr4)
print(arr_true_false)
arr4.name='population of cities'
print(arr4)
arr4.index.name='cities vs population'
print(arr4)
arr4.index=['Gedebey','Qusar','Lerik']
print(arr4)
'''

#data={'Names':['Namiq','Ayxan','Ali','Ilkin','John'],
#      'Ages':[21,17,21,23,22],
#      'Cities':['Baku','Baku','Ganja','Agdam','Toronto']}
#
#dataframe1=pd.DataFrame(data)
#dataframe1.index.name='Info about students'
##print(dataframe1)
##print(dataframe1.head(2))
#frame2=pd.DataFrame(data,index=['one','two','three','four','five'])
##print(frame2)
##print(frame2.index)
##print(frame2['Names'])
##print(frame2.loc['two'])
#numbers=[91,92,93,94,95]
#frame2['Score']=[i for i in numbers]
#frame2.columns=['Name','Age','City','Score']
##print(frame2)
#
#values=pd.Series([4.5,5,2.5,3.4,4.1],index=['one','two','three','four','five'])
#frame2['Star']=values
##print(frame2)
#frame2['isGraduated']=frame2.Score==99
##print(frame2)
#del frame2['isGraduated']
#print(frame2.columns)
#print(frame2.values)


numbers=[x for x in range(10)]
obj=pd.Series(numbers,index=['d','b','f','e','a','g','j','c','l','i'])

#print(obj.sort_index())
frame=pd.DataFrame(np.arange(9).reshape((3,3)),index=['1st row','2nd row','3rd row'],columns=['a','b','c'])
#print(frame)

obj2=pd.Series([1,2,3,4,5,6,7])
print(obj2.rank())