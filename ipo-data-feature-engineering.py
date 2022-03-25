import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrix

%matplotlib inline


df = pd.read_csv('~/machine-learning-end-to-end/IPO-market-forecast/ipo-data.csv')

df.head()
df = df.drop(df.columns[0],axis=1)

df['Opening Gap % Chg'] = (df['Opening Price'] - df['Offer Price']) / df['Offer Price']

def get_mgr_count(x):
    return len(x.split('/'))

df['Mgr Count'] = df['Managers'].apply(get_mgr_count)

df.groupby('Mgr Count')['1st Day Open to Close % Chg'].mean().to_frame().style.bar(align='mid',color=['#d65f5f','#5fba7d'])

df['Lead Mgr'] = df['Managers'].apply(lambda x:x.split('/')[0])

df['Lead Mgr'].unique()

y = df['1st Day Open to Close % Chg'].apply(lambda x: 1 if x > 0.025 else 0)

X = dmatrix("Q('Opening Gap % Chg') + C(Q('Month'), Treatment) + C(Q('Day of Week'), Treatment) + Q('Mgr Count') + Q('Lead Mgr') + Q('Offer Price') + C(Q('Star Rating'), Treatment)", df, return_type="dataframe")

df.to_csv('final_data.csv')

X.columns
X.dtypes

y.to_csv('target.csv')
X.to_csv('features.csv')
