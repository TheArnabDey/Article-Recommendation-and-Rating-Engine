from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
df2 = pd.read_csv('test.csv')
df1 = pd.read_csv('train.csv')
df3 = pd.read_csv('article.csv')
df3.iloc[:,1] = df3.iloc[:,1].fillna(df3.iloc[:,1].mean())
df1 = pd.merge(df1, df3, on='Article_ID')
df1 = df1.drop("ID", axis=1)
df = pd.read_csv('train.csv')
dfi = df.groupby(['Article_ID'])['Rating'].mean()
dfi = pd.DataFrame(dfi)
dfi.columns = ["MeanRating"]
dfi["Article_ID"] = dfi.index
df1 = pd.merge(df1, dfi, on='Article_ID')
df2 = pd.merge(df2, df3, on='Article_ID')
df2 = pd.merge(df2, dfi, on='Article_ID', how='left')
df2 = df2.fillna(df2.iloc[:,6].mean())
df1 = df1.drop("User_ID", axis=1)
df1 = df1.drop("Article_ID", axis=1)
df2 = df2.drop("User_ID", axis=1)
df2 = df2.drop("Article_ID", axis=1)
clf = GradientBoostingRegressor()
print df1.iloc[:,1:]
print df2.info()
clf.fit(df1.iloc[:,1:], df1.iloc[:,0])
pred = clf.predict(df2.iloc[:,1:])
df2["Rating"] = pred
df2 = df2.drop("VintageMonths", axis=1)
df2 = df2.drop("NoOfArticlesBySameAuthor", axis=1)
df2 = df2.drop("NoOfArticlesInSameCat", axis=1)
df2 = df2.drop("MeanRating", axis=1)
df2.to_csv('Submission.csv', index= False)