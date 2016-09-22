'''
import json
import pandas as pd
from glob import glob

def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob

json_filename="yelp_train_academic_dataset_business.json"

for json_filename in glob('*.json'):
    csv_filename = '%s.csv' % json_filename[:-5]
    #print ('Converting %s to %s') % (json_filename, csv_filename)
    df = pd.DataFrame([convert(line) for line in file("yelp_train_academic_dataset_business.json")])
    df.to_csv(csv_filename, encoding='utf-8', index=False)

       df = pd.DataFrame([convert(line) for line in file(json_filename)])


import json
import nltk
import collections

from pandas.tools.plotting import scatter_matrix


import json
import pandas as pd
from glob import glob
import matplotlib as plt
import pickle



import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


print pd.read_csv("yelp_train_academic_dataset_business.csv")
print (pd.read_csv("yelp_train_academic_dataset_business.csv"))

df = pd.read_csv("yelp_train_academic_dataset_business.csv")
'''
#//anaconda/lib/python3.5/site-packages/IPython/core/interactiveshell.py:2723: DtypeWarning: Columns (6,11,27,30) have mixed types. Specify dtype option on import or set low_memory=False.
#  interactivity=interactivity, compiler=compiler, result=result)
'''
dataset_array = df.values
print(dataset_array.dtype)
print(dataset_array)

list(df.columns.values)

print 'After printing data set'

#labels = list(df.columns.values)
labels = list(df.stars)
training = np.asmatrix(df)
labels = np.asarray([str(v) for v in labels])
le = preprocessing.LabelEncoder()
indices = np.arange(training.shape[0])
encoded = le.fit_transform([str(v) for v in labels]) 

print 'Setting up data set split'

X_train, X_test, y_train, y_test, indices_train, indices_test  = train_test_split(training, encoded, indices, test_size=0.3)

print 'Post data split'

class PosNegEstimator(sk.base.BaseEstimator, sk.base.RegressorMixin):
    """
    Predict mean values based on whether there are more positive or negative features.
    
    Yes, this will do a terrible job.
    """
    def __init__(self):
        self.pos_mean = 0
        self.neg_mean = 0
    
    def fit(self, X, y):
        pos_rows = (X > 0).sum(axis=1) > X.shape[1]/2
        self.pos_mean = y[pos_rows].mean()
        self.neg_mean = y[~pos_rows].mean()
        return self
    
    def predict(self, X):
        pos_rows = (X > 0).sum(axis=1) > X.shape[1]/2
        y = np.zeros(X.shape[0])
        y[pos_rows] = self.pos_mean
        y[~pos_rows] = self.neg_mean
        return y

# Make A Transformer
class CenteringTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Centers the features about 0
    """
    def fit(self, X, y=None):
        self.means = X.mean(axis=0)
        return self
    
    def transform(self, X):
        return X - self.means

ct = CenteringTransformer()
ct.fit(X_train.city)
X_test.mean(), ct.transform(X_test).mean()

pne = PosNegEstimator()
pne.fit(X_train, y_train)
pne.predict(X_test)
pne.score(X_test, y_test)     

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
'''
# -*- coding: utf-8 -*-
"""

"""


import json
import nltk
import collections

from pandas.tools.plotting import scatter_matrix


import json
import pandas as pd
from glob import glob
import matplotlib as plt
import pickle



import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#DEFINE BASE BARIABLES LIKE THE FILE NAME
json_list = []
json_filename = ''

# OLD CODE
#for line in json_filename:
#    json_list.append(json.loads(line))

# READ FILE IN
df = pd.DataFrame([convert(line) for line in file(json_filename)])

#dfrestaurants = pd.DataFrame.from_records(json_list)

#QUESTOIN 1 ANSWER

df = pd.read_csv("yelp_train_academic_dataset_business.csv")


todict = df.groupby(['city']).mean()
dictionary = todict[['stars']]
#CHANGE DATAFRAME INDEX NAME
dictionary.index.name = 'city'
# Then pickle this if wanted.....
import pickle
dictionary.to_pickle("q1.pkl")
#new_df = pd.read_pickle("q1.pkl")
#favorite_color = { "lion": "yellow", "kitty": "red" }
#pickle.dump( favorite_color, open( "save.p", "wb" ) )

### QUESTION 2 LAT LONG RESEARCH
print 'Setting up data set split'

training = df[['latitude','longitude']]

X_train, X_test, y_train, y_test = train_test_split(training, df.stars, test_size=0.3, random_state=42)

parameters = {'n_neighbors' : [1,5,10,15,20]}
neigh =  KNeighborsRegressor()
modelneigh = grid_search.GridSearchCV(neigh, parameters) 

#neigh = KNeighborsRegressor(n_neighbors=25)
modelneigh.fit(X_train,y_train)
pred = modelneigh.predict(X_test)
results = modelneigh.score(X_test, y_test)

### RANDOM FORREST MODEL

#parameters2 = {'n_estimators' : [1,5,10,15,20]}
#neigh2 =  RandomForestRegressor()
#modelneigh2 = grid_search.GridSearchCV(neigh2, parameters2) 

##neigh2 = RandomForestRegressor(n_estimators=25)
#modelneigh2.fit(X_train,y_train)
#pred2 = neigh2.predict(X_test)
#results2 = neigh2.score(X_test, y_test)

### NOW MAKE THE PIPELINE

from sklearn import pipeline

class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, df, y=None, cols=['latitude', 'longitude']):
       return df[cols]

   def fit(self, df, y=None):
       return self

q2_pipe = pipeline.Pipeline([
  ('column_select', ColumnSelectTransformer()),
  #('parameters', {'n_neighbors' : [1,5,10,15,20]}),
  #('knn', KNeighborsRegressor()),
  ('modelneigh', grid_search.GridSearchCV(neigh, parameters))
  ])
  
  guess = pipeline.Pipeline(
    [( 'column_select', ColumnSelectTransformer())] +
    [( 'modelneigh', grid_search.GridSearchCV(neigh,parameters) )] +
    []
)
q2_pipe.fit(X_train, y_train)
print q2_pipe.score(X_test, y_test)
### SCORE IS GOOD
#0.00190566039156


#pipeline = Pipeline([('column_select', ColumnSelectTransformer()), ('knn', neighbors.KNeighborsRegressor())])


#Q3:
'''use category column'''

'''df[['categories']]'''

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

todict = df.groupby(['categories']).mean()
todict= todict.to_dict().values()
dictionary = todict[['stars']]
#CHANGE DATAFRAME INDEX NAME
dictionary.index.name = 'categories'
# Then pickle this if wanted.....
import pickle
dictionary.to_pickle("q3.pkl")

training = df[['categories']]
#y = df['stars'] #?????? this is in the line below, as df.stars, so not needed


todict = df.groupby(['categories']).mean()
todict=todict[['stars']]
todict= todict.to_dict()


import math

categories=[]
stars=[]


todict2=[]
for k,v in todict.items():
    #print('v: '+str(v))
    for k2,v2 in v.items():
        if math.isnan(v2)==False:
            categories.append(k2)
            stars.append(v2)
categories=np.array(categories)
stars=np.array(stars)
categories=categories.reshape((5561,1))
stars=stars.reshape((5561,1))

                #todict2.append({'category':k2,'star':v2})

# instantiate linear regression
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(categories,stars)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)

# pair the feature names with the coefficients
zip(cols, linreg.coef_)



vec = DictVectorizer(sparse=False)
vectorized = vec.fit_transform(todict2)

X_train, X_test, y_train, y_test = train_test_split(df, df.stars, test_size=0.2, random_state=42)


class CustomTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, df, y=None, cols=['categories']):
       return df[cols]

   def fit(self, df, y=None):
       return self

pipeline = Pipeline([('category', CategoriesTransformer()), ('vectorizer', DictVectorizer()), ('lr', LinearRegression())])

pipeline.fit(X_train, y_train)


#todict.to_dict().values()


vectorizer=DictVectorizer( sparse=False ).fit_transform(todict)

pipeline = Pipeline([('vectorizer', DictVectorizer()), ('lr', LinearRegression())])

vec = DictVectorizer(sparse=False)



X_train, X_test, y_train, y_test = train_test_split(training, df.stars, test_size=0.3, random_state=42)


#put into dict
#x_cat_train = cat_train.T.to_dict().values()
#x_cat_test = cat_test.T.to_dict().values()

# vectorize
#vectorizer = DV( sparse = False )
#vec_training = vectorizer.fit_transform(training)

#vec_test = vectorizer.transform( X_test)


class CustomTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, df, y=None, cols=['categories']):
       return df[cols]

   def fit(self, df, y=None):
       return self

q3_pipe = pipeline.Pipeline([
  ('column_select', CustomTransformer()),
#  ('parameters', {'n_neighbors' : [1,5,10,15,20]}),
#  ('knn', KNeighborsRegressor()),
 #('tfidf', TfidfVectorizer()),
    ( 'dv', DictVectorizer(sparse=False) )])
  
#q3_pipe.fit(X_train, y_train)
#print q3_pipe.score(X_test, y_test)

# instantiate linear regression
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)

# pair the feature names with the coefficients
zip(cols, linreg.coef_)


# make predictions on the testing set
y_pred = linreg.predict(X_test)

#compute RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

class CategoriesTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, df, y=None):
       dfcats = df['categories']
       dictcats = [{'categories': ",".join(cat)} for cat in dfcats]
       return dictcats

#dictcats=[]
#catstring=''
#for cat in dfcats:
##           catstring=catstring+', '+str(cat)#dictcats.append({'categories': ', '.join(cat)})
#dictcats=[{'categories': str(catstring)}]

   def fit(self, df, y=None):
       return self

pipeline = Pipeline([('category', CategoriesTransformer()), ('vectorizer', DictVectorizer()), ('lr', LinearRegression())])

pipeline.fit(dfcats, df.stars)


categoryModel = 
Pipeline([
           ('col', columnSelectTransformer('categories')), 
           ('cat', categoryTransformer()),
           ('dictVect', DictVectorizer(sparse=False)),
           ('linReg',linear_model.LassoCV(cv=5,random_state=7))
              ])
categoryModel.fit(X,y)

#Q4

from sklearn.preprocessing import LabelEncoder

'attributes': {
        'accepts_credit_cards',
        'accepts_insurance',
        'ages_allowed',
        'alcohol',
        'attire',
        'by_appointment_only',
        'byob',
        'byob/corkage',
        'caters',
        'coat_check',
        'corkage',
        'delivery',
        'dogs_allowed',
        'drive-thru',
        'good_for_dancing',
        'good_for_groups',
        'good_for_kids',
        'happy_hour',
        'has_tv',
        'noise_level',
        'open_24_hours',
        'order_at_counter',
        'outdoor_seating',
        'price_range',
        'smoking',
        'take-out',
        'takes_reservations',
        'waiter_service',
        'wheelchair_accessible',
        'wi-fi',
        'ambience': {
            'casual',
            'classy',
            'divey',
            'hipster',
            'intimate',
            'romantic',
            'touristy',
            'trendy',
            'upscal'
        },

#???????????????
class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, df, y=None, cols=['attributes']):
       return df[cols]

   def fit(self, df, y=None):
       return self

#??????????????
class FlattenTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):

   def __init__(self):
       pass

   def transform(self, df, y=None, cols=['attributes']):
       return df[cols]

   def fit(self, df, y=None):
       return self


Q4_pipe = pipeline.Pipeline([
 ('columns', ColumnSelectTransformer('attributes')),
 ('flatten', FlattenTransformer()),
 ('dictvector', DictVectorizer()),
 ('linear', linear_model.LinearRegression(fit_intercept=True))
 ])

dill.settings['recurse'] = True


a = df(columns = record.keys())   
   for key in record.keys():
       a.loc[0,key] = record[key]

alist = list()
   alist.append(record)
   a = pd.DataFrame.from_dict(alist)
   return att_model.predict(a)[0][0]


number=LabelEncoder()
train['sex']=number.fit_transform(train['sex'].astype('str'))
{'Attire': 'casual',
     'Accepts Credit Cards': True,
     'Ambience': {'casual': False, 'classy': False}}

   {'Attire_casual' : 1,
     'Accepts Credit Cards': 1,
     'Ambience_casual': 0,
     'Ambience_classy': 0 }
'''
## attribute_knn_model
Venues have (potentially nested) attributes:
```
    {'Attire': 'casual',
     'Accepts Credit Cards': True,
     'Ambience': {'casual': False, 'classy': False}}
```

Categorical data like this should often be transformed by a One Hot Encoding.
For example, we might flatten the above into something like this:

```
    {'Attire_casual' : 1,
     'Accepts Credit Cards': 1,
     'Ambience_casual': 0,
     'Ambience_classy': 0 }
```

Build a custom transformer that flattens attributes and feed this into
`DictVectorizer`.  Feed it into a (cross-validated) linear model (or something
else!)




## city_model
'''
The venues belong to different cities.  You can image that the ratings in some
cities are probably higher than others and use this as an estimator.

Build an estimator that uses `groupby` and `mean` to compute the
average rating in that city.  Use this as a predictor.

**Question:** In the absence of any information about a city, what score would
you assign a restaurant in that city?

use city 'city' and mean stars 'stars' as a predictor (Estimator)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)

# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)
:
# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))

**Note:** `def city_model` etc. takes an argument `record`.
'''

## lat_long_model
'''You can imagine that a city-based model might not be sufficiently fine-grained.
For example, we know that some neighborhoods are trendier than others.  We
might consider a K Nearest Neighbors or Random Forest based on the latitude
longitude as a way to understand neighborhood dynamics.

You should implement a generic `ColumnSelectTransformer` that is passed which
columns to select in the transformer and use a non-linear model like
`sklearn.neighbors.KNeighborsRegressor` or
`sklearn.ensemble.RandomForestRegressor` as the estimator (why would you choose
a non-linear model?).  Bonus points if you wrap the estimator in
`grid_search.GridSearchCV` and use cross-validation to determine the optimal
value of the parameters.

'latitude', 'longitude'
>>> from sklearn.neighbors import NearestNeighbors
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
>>> distances, indices = nbrs.kneighbors(X)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.predict([[Xnew]])
y_pred = knn.predict(X)

'''

