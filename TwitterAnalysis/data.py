import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('TwitterAnalysis/Tweets.csv')


data['negativereason'].fillna('Unknown')
data['negativereason_confidence'].fillna(data['negativereason_confidence'].mean())
data['airline_sentiment_gold'].fillna('Unknown')
data['negativereason_gold'].fillna('Unknown')
data['tweet_location'].fillna('Unknown')
data['user_timezone'].fillna('Unknown')
data.drop('tweet_coord',axis=1)
data['tweet_created'] = pd.to_datetime(data['tweet_created'],format='mixed',errors='coerce')
data['tweet_created']=data['tweet_created'].dt.tz_localize(None)

#print('Number of rows-{},columns-{}'.format(data.shape[0],data.shape[1]))
#print(data.info())