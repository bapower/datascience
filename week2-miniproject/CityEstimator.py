import pandas as pd

class CityEstimator(object):
	def __init__(self):
		self.data = pd.DataFrame(columns=['city', 'stars'])

	def fit(self, df):
		self.data = df.groupby(['city']).mean().reset_index()
		self.data = self.data[['city','stars']]
		return self

	def predict(self, X):
		stars = self.data.loc[self.data.city == X, 'stars'] 
		if (stars.empty) :
				return False
		else :
			return stars.iloc[0]
