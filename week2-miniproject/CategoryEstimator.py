import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

class CategoryEstimator(object):
	def __init__(self):
		self.data = pd.DataFrame(columns=['categories', 'stars'])
		self.clf = LinearRegression()
		self.vectorizer = DictVectorizer(sparse=False)

	def dictVectorize(self, df):
		self.data = df.groupby(['categories']).mean().reset_index()
		self.data = self.data[['categories','stars']]
		self.dict = self.data.to_dict().values()
		self.vector = self.vectorizer.fit_transform(self.dict)

		self.X = self.vector[0]
		self.y = self.vector[1]

		return self.vector

	def fit(self, df):
		self.vector = self.dictVectorize(df)
		self.clf.fit(self.X.reshape(len(self.X), 1), self.y)
		return self

	def predict(self, X):
		return self.clf.predict([X])

	def score (self):
		return self.clf.score(self.X.reshape(len(self.X), 1), self.y)
