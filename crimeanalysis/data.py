import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob


data = pd.read_csv('crimeanalysis/CriminalAnalysis.csv')
if data.isnull().values.any() == True:
    print('There is missing value')
else:
    print('There is no missing value')
