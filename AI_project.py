import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from scipy import stats
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import norm, skew #for some statistics



from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from mlxtend.regressor import StackingRegressor
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={"figure.figsize":(8, 8)})

housing = pd.read_csv('data.csv')

housing.head(5)

housing.shape

housing.info()

"""# Data Preprocessing

# Converting data into desired type
"""

housing['date'] = pd.to_datetime(housing['date'])

housing['price']     = housing['price'].astype('int64')
housing['bedrooms']  = housing['bedrooms'].astype('int64')
housing['floors']    = housing['floors'].astype('int64')
housing['street']    = housing['street'].astype('string')
housing['city']      = housing['city'].astype('string')
housing['statezip']  = housing['statezip'].astype('string')
housing['country']   = housing['country'].astype('string')

"""# Adding a new column"""

housing.insert(1, "year", housing.date.dt.year)
housing.info()

housing.drop_duplicates()

"""# Removing rows having price=0"""

(housing.price == 0).sum()

housing['price'].replace(0, np.nan, inplace = True)
housing.dropna(inplace=True)

housing.shape

housing.describe().T

housing.head()

housing.isnull().sum()

housing.head(5)

housing.nunique(axis = 0)

"""# Adding a new column age"""

housing['age'] = housing['year'] - housing['yr_built']

"""# How price is distributed"""

housing['price'].hist(bins=100)

# Set axis labels and plot title
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')

# Show the plot
plt.show()

"""# Removing rows with outliers"""

(housing['price'] > 0.3e7).sum()

housing = housing[~(housing['price'] > 0.3e7)]

"""# Price distribution after removal of outliers"""

housing['price'].hist(bins=100)

# Set axis labels and plot title
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')

# Show the plot
plt.show()

sns.distplot(housing['price'],color="red",kde=True,fit=norm)

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(housing['price'], plot=plt)
plt.show()

housing['price'] = np.log1p(housing['price'])

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(housing['price'], plot=plt)
plt.show()

sns.distplot(housing['price'],color="red",kde=True,fit=norm)

sns.heatmap(housing.corr(), annot=True)
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()

housing.hist()

"""# Applying one hot encoding to city and then reducing dimension using pca"""

from sklearn.decomposition import PCA
X = pd.get_dummies(housing.city, prefix='City')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
housing['city_pca1'] = X_pca[:, 0]
housing['city_pca2'] = X_pca[:, 1]

sns.heatmap(housing.corr(), annot=True)
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()

housing.columns

"""# Dropping unnecessary variables"""

data = housing.drop(['date', 'street', 'statezip', 'country','year','city','age'], axis = 1)

data.head()

data.shape

"""# Making train-test split"""

x = data.drop("price", axis=1)
y = pd.DataFrame(data["price"])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state =42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = x.columns)
X_train.head(10)

X_test = pd.DataFrame(scaler.transform(X_test), columns = x.columns)
X_test.head(10)

X_train.shape

lr = LinearRegression()
lr.fit(X_train, Y_train).score(X_test, Y_test)

Y_pred = lr.predict(X_test)

print("RMSE : ", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


#2. Support Vector Regressor RBF
svr = SVR(kernel='rbf')

#2. Support Vector Regressor RBF
svrlin = SVR(kernel='linear')

lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)




#5. Random Forest Regressor
rf = RandomForestRegressor()

#6. Linear Regressor
lr = LinearRegression()

#7. Ridge
ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#8. Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

#9. ElasticNet
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

regressor=[gbr,xgb,lr,ENet,svrlin,lgb,ridge,lasso]

model = StackingRegressor(regressors=regressor, 
                           meta_regressor=svr)

model.fit(X_train, Y_train)

model.score(X_test, Y_test)

Y_pred = model.predict(X_test)

r2_score(Y_test,Y_pred)

print("R2 Score : ", metrics.r2_score(Y_test, Y_pred))

print("RMSE : ", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))