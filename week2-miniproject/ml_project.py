import pandas as pd
from CityEstimator import CityEstimator
from LocationEstimator import LocationEstimator
from CategoryEstimator import CategoryEstimator

df = pd.read_csv("yelp_train_academic_dataset_business.csv", low_memory=False)

CityEstimator = CityEstimator().fit(df)
prediction = CityEstimator.predict('Verona')
print(prediction)


LocationEstimator = LocationEstimator().fit(df)
prediction = LocationEstimator.predict([1, -15])
score = LocationEstimator.score()
print(score)


CategoryEstimator = CategoryEstimator().fit(df)
prediction = CategoryEstimator.predict(2)
score = CategoryEstimator.score()
print(score)

