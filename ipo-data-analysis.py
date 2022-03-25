import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import seaborn as sns
%matplotlib inline

wb = xlrd.open_workbook('~/machine-learning-end-to-end/IPO-market-forecast/SCOOP-Rating-Performance.xls')

ws = wb.sheet_by_index(0)

ws.nrows
ipo_lists = []

for i in range(36,ws.nrows):
    if isinstance(ws.row(i)[0].value, float):
        ipo_lists.append([x.value for x  in ws.row(i)])
    else:
        print(i, ws.row(i))
        
len(ipo_lists)

df = pd.DataFrame(ipo_lists)
df.head()

df.columns = ['Date', 'Company', 'Ticker', 'Managers', \
'Offer Price', 'Opening Price', '1st Day Close',\
'1st Day % Chg', '$ Chg Open', '$ Chg Close',\
'Star Rating', 'Performed']
    
def to_date(x):
    return xlrd.xldate.xldate_as_datetime(x, wb.datemode)

df['Date'] = df['Date'].apply(to_date)

df['Year'], df['Month'], df['Day'], df['Day of Week'] = df['Date'].dt.year, df['Date'].dt.month, df['Date'].dt.day, df['Date'].dt.weekday

year_cnt = df.groupby('Year')[['Ticker']].count()
year_cnt

df.drop(df[df['Offer Price'] < 5].index, inplace=True)
df.reset_index(drop=True, inplace=True)
df.dtypes

df['1st Day % Chg'] = df['1st Day % Chg'].astype(float)




a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot(x='Year',data=df)

summer_by_year = df.groupby('Year')['1st Day % Chg'].describe()
summer_by_year

sns.barplot(x=summer_by_year.index,y='mean',data=summer_by_year)

df['1st Day Open to Close % Chg'] = ((df['1st Day Close'] - df['Opening Price'])/df['Opening Price'])
df['1st Day Open to Close % Chg'].describe()

sns.barplot(x="Year",y='1st Day Open to Close % Chg',data=df)

df['1st Day Open to Close $ Chg'] = (df['1st Day Close'] - df['Opening Price'])
df[df['Year'] == 2018].sum()
df[df['Year'] == 2018]['1st Day Open to Close $ Chg'].describe()

sns.histplot(df[df['Year'] == 2018]['1st Day Open to Close $ Chg'])
df.head()
df.to_csv('ipo-data.csv')
