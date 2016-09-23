import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

class LocationEstimator(object):
	def __init__(self):
		self.data = pd.DataFrame(columns=['stars', 'latitude', 'longitude'])
		self.neighbors = KNeighborsRegressor(n_neighbors=15, weights='distance')
		self.X = []
		self.y = []

	def fit(self, df):
		self.data = df[['stars', 'latitude', 'longitude']]
		for index, row in self.data.iterrows():
			point = [row['latitude'], row['longitude']]
			self.X.append(point)
			self.y.append(row['stars'])
		self.neighbors.fit(self.X, self.y)
		return self

	def predict(self, X):
		return self.neighbors.predict([X])

	def score (self):
		return self.neighbors.score(self.X,self.y)
