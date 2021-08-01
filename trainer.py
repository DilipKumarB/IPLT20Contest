import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OrdinalEncoder
import joblib

data = pd.read_csv('myPreprocessed.csv')

venue_encoder = OrdinalEncoder(handle_unknown='ignore')
batting_encoder = OrdinalEncoder(handle_unknown='ignore')
bowling_encoder = OrdinalEncoder(handle_unknown='ignore')

# venue_encoder = OrdinalEncoder(cols=['venue'])
# batting_encoder = OrdinalEncoder(cols=['batting_team'])
# bowling_encoder = OrdinalEncoder(cols=['bowling_team'])

# encoder = ce.OrdinalEncoder(cols = ['venue','batting_team','bowling_team'] )

data[['venue']] = venue_encoder.fit_transform(data[['venue']])
data[['batting_team']] = batting_encoder.fit_transform(data[['batting_team']])
data[['bowling_team']] = bowling_encoder.fit_transform(data[['bowling_team']])

X = data.drop('total_runs', axis=1)
y = data['total_runs']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# from sklearn.linear_model import Lasso
# from sklearn.model_selection import GridSearchCV
#
# lasso = Lasso()
# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40]}
# lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
#
# lasso_regressor.fit(X_train, y_train)
# print(lasso_regressor.best_params_)
# print(lasso_regressor.best_score_)
#
# prediction = lasso_regressor.predict(X_test)
#
# from sklearn import metrics
# import numpy as np
#
# print('MAE:', metrics.mean_absolute_error(y_test, prediction))
# print('MSE:', metrics.mean_squared_error(y_test, prediction))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

from sklearn.svm import SVR
# regressor = SVR(kernel='linear',degree=1)
#regressor = SVR(kernel = 'rbf',epsilon = 1.0)

#regressor.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000],
              'gamma' : [1,0.1,0.01,0.01,0.001,0.0001],
              'kernel' : ['rbf']
             }
grid = GridSearchCV(SVR(),param_grid,refit = True, verbose = 3)
grid.fit(X_train, y_train)

regressor = grid

joblib.dump(regressor, 'regression_model.joblib')
joblib.dump(venue_encoder, 'venue_encoder.joblib')
joblib.dump(batting_encoder, 'batting_encoder.joblib')
joblib.dump(bowling_encoder, 'bowling_encoder.joblib')

# print(lasso_regressor.score(X_test, y_test))
