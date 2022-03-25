import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

%matplotlib inline

y = pd.read_csv('~/machine-learning-end-to-end/IPO-market-forecast/target.csv')
X = pd.read_csv('~/machine-learning-end-to-end/IPO-market-forecast/features.csv')

len(X.columns)
pd.get_dummies(X)

y = y.drop(y.columns[0],axis=1)
X = X.drop(X.columns[0],axis=1)

pd.DataFrame(X.columns).apply(lambda x : x if x.startswith('C'))

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = LogisticRegression()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_true = pd.Series(y_test)

y_pred
y_true['y_hat'] = y_pred

y_true.columns = ['y_true','y_hat']
y_true['correct'] = y_true.apply(lambda x: 1 if x['y_true'] == x['y_hat'] else 0, axis=1)
y_true

y_true['y_true'].value_counts(normalize=True)
y_true['correct'].value_counts(normalize=True)

df = pd.read_csv('~/machine-learning-end-to-end/IPO-market-forecast/final_data.csv')

results = pd.merge(df[['1st Day Open to Close $ Chg']], y_true, left_index=True, right_index=True)
results

results['1st Day Open to Close $ Chg'].describe()

#ipo buys
results[results['y_hat']==1]['1st Day Open to Close $ Chg'].describe()

fv = pd.DataFrame(X_train.columns, clf.coef_.T.reshape(360,)).reset_index()
fv
fv.columns = ['Coef', 'Feature']
fv.sort_values('Coef', ascending=0).reset_index(drop=True)
fv[fv['Feature'].str.contains('Day')]
