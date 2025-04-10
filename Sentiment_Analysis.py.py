
import pandas as pd
df=pd.read_csv("IMDB Dataset.csv")

df.head()

R=df["review"]

R

S=df["sentiment"]

S

for index,row in df.iterrows():
  print(index,"Review:",row["review"],"Sentiment:",row["sentiment"])

some=df[5:10]



some

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import  CountVectorizer

#y=df['sentiment']
train_x, test_x, train_y, test_y = model_selection.train_test_split(df['review'], df['sentiment'],test_size=0.33, random_state=42)



vectTrain_X = CountVectorizer(input='content',decode_error='ignore',analyzer='word',ngram_range=(1,3)).fit(train_x)
X_train_vectorized = vectTrain_X.transform(train_x)

X_Test_vectorized = vectTrain_X.transform(test_x)

lr=LogisticRegression()
lr.fit(X_train_vectorized ,train_y)



y_prediction=lr.predict(X_Test_vectorized)

y_prediction

from sklearn.metrics import classification_report
print(classification_report(test_y, y_prediction))