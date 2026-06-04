### Project Overview
A project where we'll find and clean a large dataset, build and train various ML models, and then report the training performance of such models


### Data Sources
For the purposes of my models, I wanted my dataset to have no less than 10,000 data samples and at least 10 features for each sample. 

Because I am an avid cycler, a supporter of public transportation, and a prolific user of bike sharing programs, I decided to find some data on a bikesharing application.

I found my set from the University of California Irvine's Machine Learning Repository: https://archive.ics.uci.edu/. This set tracks the Capital bikeshare system between 2011 and 2012, including the hourly and daily count of rental bikes. It has over 17000 samples and 13 features per. 

Link: [Capital Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

### Tools:
Numpy & Pandas --> Standard data manipulation
Matplotlib.pyplot --> Visualizations
Sklearn.model_selection --> train_test_split
Sklearn.preprocessing --> StandardScaler, OneHotEncoder

### Phase 1 --> Cleaning and Preprocessing

Starting in the **proj3_data_preprocess.ipynb** file, we loaded our dataset into our file (Note: I chose to use the hourly set from the repository. There *IS* also a set that tracks by day that we will not be using in this project)
```python
full_data = pd.read_csv('hour.csv')
full_data.describe()
``` 
<img width="1994" height="565" alt="image" src="https://github.com/user-attachments/assets/2183d26a-3bae-4299-810d-191a918a7a1b" /> 

Looking at the datatypes with full_data.dtypes: 
<img width="344" height="636" alt="image" src="https://github.com/user-attachments/assets/fab2f5f3-5022-4bc9-8224-4b9befa2fb9c" />

The first and best change we can make is making the date variable (dteday) into a datetime such that we can actually do math with it. Like so:
```python
full_data['dteday'] = pd.to_datetime(full_data['dteday'])
```

Then we'll check for null values anywhere in the set, as well as any duplicates:
```python
full_data.isnull().sum()
``` 
<img width="259" height="601" alt="image" src="https://github.com/user-attachments/assets/4806141d-e324-4821-98dc-737e740f1715" />

```python
full_data.duplicated().sum() == 0
```
Returns TRUE!

Nice, we have none. It seems like this set is already processed very well!


Our next biggest step is to encode our categorical variables:
```python
full_data.corr()
```
<img width="2491" height="1111" alt="image" src="https://github.com/user-attachments/assets/6b6d7717-e05b-4c75-aca9-d83b8c4f3f4d" /> 

```python
target_corr = full_data.corrwith(full_data['cnt'])
print(target_corr.sort_values(ascending=False))
```
<img width="347" height="619" alt="image" src="https://github.com/user-attachments/assets/9b40b10d-f269-4073-9554-365ae7097933" /> 

And we'll also create one of our own variables, a cross between hour and temperature (hr_temp), to better account for the crossover in context between them. This may turn out to be useless to us, but when choosing our best features this may come to benefit us.

```python
full_data["hr_temp"] = full_data["hr"] * full_data["temp"]
print(full_data.corr()["cnt"].sort_values(ascending=False))
```
<img width="360" height="42" alt="image" src="https://github.com/user-attachments/assets/ac20b7c2-e022-42a2-84c6-c030496521de" /> 


Annnnnnd our data processing is complete. Typically there'd be some more we can do, but this data came from a very organized and reputable source, so it's no surprise it's mostly usable already.

```python
full_data
```
Yields: 
<img width="2149" height="796" alt="image" src="https://github.com/user-attachments/assets/6b251665-3223-4f7b-b32c-419240836734" /> 

Let's separate that into a training, testing, and validation set, and then export into csv files:

```python
train_vals, test_data = train_test_split(full_data, test_size=0.2)
train_data, validation_data = train_test_split(train_vals, test_size=0.25)

print(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}, Test size: {len(test_data)}")
```
<img width="852" height="56" alt="image" src="https://github.com/user-attachments/assets/95d3fc9d-bf81-45d9-8d84-68ff82be71cb" /> 

```python
train_data.to_csv("cleaned_train_data.csv")
validation_data.to_csv("cleaned_validation_data.csv")
test_data.to_csv("cleaned_test_data.csv")
```

### Phase 2 --> Building and Training Various Machine Learning Models:

Now we'll be working in the **proj3_machine_learning.ipynb** file. We'll need all the same imports as earlier, aside from the OneHotEncoder. 

We'll start by loading in our data, and we'll initially remove our useless or highly correlated features:

```python
train_data = pd.read_csv('cleaned_train_data.csv')
validation_data = pd.read_csv('cleaned_validation_data.csv')
test_data = pd.read_csv('cleaned_test_data.csv')

X_train = train_data.drop(['cnt', 'casual', 'registered', 'dteday', 'Unnamed: 0', 'instant', 'yr', 'mnth', 'atemp'], axis=1, inplace=False)
y_train = train_data['cnt']

X_test = test_data.drop(['cnt', 'casual', 'registered', 'dteday', 'Unnamed: 0', 'instant', 'yr', 'mnth', 'atemp'], axis=1, inplace=False)
y_test = test_data['cnt']

X_valid = validation_data.drop(['cnt', 'casual', 'registered', 'dteday', 'Unnamed: 0', 'instant', 'yr', 'mnth', 'atemp'], axis=1, inplace=False)
y_valid = validation_data['cnt']
```

Let's make sure our load looks good:
```python
X_train
```
Yields: 
<img width="1354" height="804" alt="image" src="https://github.com/user-attachments/assets/4a76368c-593a-4f16-9379-a8eabb72a9d6" /> 
Which looks great.

We can also ensure the shapes of our X sets are the same:
```python
X_train.shape, X_test.shape, X_valid.shape
``` 
<img width="563" height="56" alt="image" src="https://github.com/user-attachments/assets/41922ae1-e495-4e7f-97b7-7ea14f64f5b9" /> 

Perfect! Let's fit them to our scaler:

```python
scaler = StandardScaler().set_output(transform = 'pandas')

X_train = scaler.fit_transform(X_train)
X_train
```
Which now yields us: 
<img width="1610" height="788" alt="image" src="https://github.com/user-attachments/assets/74e73389-8c02-4091-a981-f19f11f09b48" /> 

```python
X_test = scaler.fit_transform(X_test)
X_valid = scaler.fit_transform(X_valid)
```
So now our variables have all been scaled and we can begin with our first machine learning model. For the following, we will be using RMSE and R^2 as our loss functions to interpret the strength of each model. At the end we'll compare to see what our best option is.

#### Linear Regression:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_predictions = linear_model.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predictions))
r2_train = r2_score(y_train, y_train_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")
``` 
<img width="573" height="48" alt="image" src="https://github.com/user-attachments/assets/f690c7fa-d5a0-4eea-945f-fbc8fa623984" /> 

```python
y_valid_predictions = linear_model.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_valid_predictions))
r2_valid = r2_score(y_valid, y_valid_predictions)
print(f"Validation RMSE: {rmse_valid:.3f}, R^2: {r2_valid:.3f}")
``` 
<img width="559" height="48" alt="image" src="https://github.com/user-attachments/assets/89589310-76f7-4906-a90f-2e8d34cf402b" /> 

Our RMSE and . Our R^2 of 0.33 indicates that our model only explains a third of the data. This is not very high, which is what'd we expect from a linear model that doesn't generalize well to stranger data points.

In addition, we'll try a couple of Kernel Tricks to our linear regression to see if we can get some better results:

First, **RBF Kernel**

``` python
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge

#RBF first:
linear_rbf = KernelRidge(kernel='rbf', alpha=0.01, gamma=0.1).fit(X_train, y_train)
linear_rbf.fit(X_train, y_train)

y_train_rbf_predictions = linear_rbf.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_rbf_predictions))
r2_train = r2_score(y_train, y_train_rbf_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")
``` 
<img width="532" height="53" alt="image" src="https://github.com/user-attachments/assets/01c24a36-186a-4964-ae3f-ab3eb108a19f" />

```python
rbf_y_valid_predictions = linear_rbf.predict(X_valid)
rbf_rmse_valid = np.sqrt(mean_squared_error(y_valid, rbf_y_valid_predictions))
rbf_r2_valid = r2_score(y_valid, rbf_y_valid_predictions)
print(f"RBF-Kernelized Linear Model Validation RMSE: {rbf_rmse_valid:.3f}, R^2: {rbf_r2_valid:.3f}")
``` 
<img width="938" height="41" alt="image" src="https://github.com/user-attachments/assets/023fa90a-9a19-4ac8-ae3c-7846df4948f2" /> 


And secondly, **Polynomial Kernel**

```python
linear_poly = KernelRidge(kernel="poly", degree=2, alpha=1.0)
linear_poly.fit(X_train, y_train)

y_train_poly_predictions = linear_poly.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_poly_predictions))
r2_train = r2_score(y_train, y_train_poly_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")

poly_y_valid_predictions = linear_poly.predict(X_valid)
poly_rmse_valid = np.sqrt(mean_squared_error(y_valid, poly_y_valid_predictions))
poly_r2_valid = r2_score(y_valid, poly_y_valid_predictions)
print(f"Poly Kernelized Linear Model Validation RMSE: {poly_rmse_valid:.3f}, R^2: {poly_r2_valid:.3f}")
``` 
<img width="956" height="83" alt="image" src="https://github.com/user-attachments/assets/901e3f87-ae79-4e26-928e-46fa17102b51" /> 

Because the RBF-Kernalized linear model performed the best of the three on the validation set, we'll apply that one to our test set: 

```python
rbf_y_test_predictions = linear_rbf.predict(X_test)
rbf_rmse_test = np.sqrt(mean_squared_error(y_test, rbf_y_test_predictions))
rbf_r2_test = r2_score(y_test, rbf_y_test_predictions)
print(f"RBF-Kernelized Linear Model Test RMSE: {rbf_rmse_test:.3f}, R^2: {rbf_r2_test:.3f}")
``` 
<img width="863" height="42" alt="image" src="https://github.com/user-attachments/assets/62f7231a-a361-4fac-9a18-38d0bd584c54" /> 

#### Support Vector Machines:
To use this we'll first have to import the proper library:
```python
from sklearn.svm import SVR
```

We'll additionally do a couple of kernel tricks here. The SVR defaults to the rbf kernel, so we will start with that.

Tune the hyperparameters below:
```python
#This takes 473 seconds
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1, 5]
}


svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svr = grid_search.best_estimator_


svr_y_valid_predictions = best_svr.predict(X_valid)
svr_rmse_valid = np.sqrt(mean_squared_error(y_valid, svr_y_valid_predictions))
svr_r2_valid = r2_score(y_valid, svr_y_valid_predictions)

print(f"Best SVR Model - Validation RMSE: {svr_rmse_valid:.4f}, R^2: {svr_r2_valid:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")
``` 
<img width="906" height="123" alt="image" src="https://github.com/user-attachments/assets/f598c2cd-4a2e-4a8c-b1eb-0174f0d0fbf0" /> 

Now we apply our tuned hyperparameters to our final RBF SVM model:
```python
gamma_param = 1 #Hyperparameters applied
C_param = 100
epsilon_param = 5

svm_rbf_model = SVR(kernel='rbf', gamma = gamma_param, C = C_param, epsilon = epsilon_param) #Prior to tuning hyperparameters: SVR (RBF Kernel) - Validation RMSE: 147.516, R^2: 0.335
svm_rbf_model.fit(X_train, y_train)

y_train_rbf_predictions = svm_rbf_model.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_rbf_predictions))
r2_train = r2_score(y_train, y_train_rbf_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")
```
Resulting in: 
<img width="557" height="52" alt="image" src="https://github.com/user-attachments/assets/45afc5dc-69bb-421c-8b06-04a37d087810" /> 

And when we apply this model to our validation set like so:
```python
svr_rbf_y_valid_predictions = svm_rbf_model.predict(X_valid)

svr_rmse_valid = np.sqrt(mean_squared_error(y_valid, svr_rbf_y_valid_predictions))
svr_r2_valid = r2_score(y_valid, svr_rbf_y_valid_predictions)

print(f"SVR (RBF Kernel) - Validation RMSE: {svr_rmse_valid:.3f}, R^2: {svr_r2_valid:.3f}")
```

We get: 
<img width="864" height="53" alt="image" src="https://github.com/user-attachments/assets/afb8f8e8-78d6-4ce7-979b-18a0260bb8c0" />


Now let's do the same with a sigmoid kernel. We'll tune our hyperparameters again just to check:
```python
#Took 283 Seconds
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1, 5]
}


svr_sig = SVR(kernel='sigmoid')
sig_grid_search = GridSearchCV(svr_sig, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
sig_grid_search.fit(X_train, y_train)

sig_best_svr = sig_grid_search.best_estimator_

print(f"Best Parameters: {sig_grid_search.best_params_}")
``` 
<img width="916" height="86" alt="image" src="https://github.com/user-attachments/assets/467d9ab2-1184-48c6-99cc-d3becb0896f9" /> 

Resulting in this training when applied: 
```python
sig_gamma_param = 0.01
sig_C_param = 100
sig_epsilon_param = 5


svm_sig_model = SVR(kernel='sigmoid', gamma = sig_gamma_param, C = sig_C_param, epsilon = sig_epsilon_param) #Prior to tuning hyperparameters: SVR (Sigmoid Kernel) - Validation RMSE: 157.342, R^2: 0.227
svm_sig_model.fit(X_train, y_train)

y_train_sig_predictions = svm_sig_model.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_sig_predictions))
r2_train = r2_score(y_train, y_train_sig_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")

svr_sig_y_valid_predictions = svm_sig_model.predict(X_valid)

svr_rmse_valid = np.sqrt(mean_squared_error(y_valid, svr_sig_y_valid_predictions))
svr_r2_valid = r2_score(y_valid, svr_sig_y_valid_predictions)

print(f"SVR (Sigmoid Kernel) - Validation RMSE: {svr_rmse_valid:.3f}, R^2: {svr_r2_valid:.3f}")
```

Yielding: 
<img width="890" height="90" alt="image" src="https://github.com/user-attachments/assets/d14c0115-1d5b-4a21-97e4-70d650ae47e2" />


Now, this model did not perform well at all, even with the "best" parameters from the grid-search. I am not sure if this is a function of my chosen search parameters or some mistake in my tuning. Either way, the RBF SVM model performe way better of the two, so we'll apply that one to our test set:

```python
svr_rbf_y_test_predictions = svm_rbf_model.predict(X_test)

svr_rmse_test = np.sqrt(mean_squared_error(y_test, svr_rbf_y_test_predictions))
svr_r2_test = r2_score(y_test, svr_rbf_y_test_predictions)

print(f"SVR (RBF Kernel) - Test RMSE: {svr_rmse_test:.3f}, R^2: {svr_r2_test:.3f}")
```
Resulting in: 
<img width="733" height="52" alt="image" src="https://github.com/user-attachments/assets/37704a55-7eb8-404f-87ea-b0449e0394e7" />





#### Decision Trees:
We'll once again tune hyperparameters with GridSearchCV:

```python
## USE GRID SEARCH HERE AS WELL TO TUNE HYPERPARAMETERS!
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeRegressor()

param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
```
<img width="1226" height="84" alt="image" src="https://github.com/user-attachments/assets/7d14ea73-9956-40f9-ad21-f20c1b808d4b" />


Then apply to training and validation sets: 
```python
#Prior to tuning hyperparameters: Decision Tree - Validation RMSE: 98.321, R^2: 0.705
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

max_depth_param = 15
min_samples_leaf_param = 10
min_samples_split_param = 10

tree_regressor = DecisionTreeRegressor(max_depth = max_depth_param, min_samples_leaf = min_samples_leaf_param, min_samples_split = min_samples_split_param)

tree_regressor.fit(X_train, y_train)

y_train_tree_predictions = tree_regressor.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_tree_predictions))
r2_train = r2_score(y_train, y_train_tree_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")

dt_y_valid_predictions = tree_regressor.predict(X_valid)

dt_rmse_valid = np.sqrt(mean_squared_error(y_valid, dt_y_valid_predictions))
dt_r2_valid = r2_score(y_valid, dt_y_valid_predictions)

mae = mean_absolute_error(y_valid, dt_y_valid_predictions)

print(f"Decision Tree - Validation MAE: {mae:.3f}")
print(f"Decision Tree - Validation RMSE: {dt_rmse_valid:.3f}, R^2: {dt_r2_valid:.3f}")
```

Yielding: 
<img width="766" height="125" alt="image" src="https://github.com/user-attachments/assets/6dcf640e-6b61-464f-9867-098b9bbfc6cf" /> 

Wow! So far that's been the best of our models. An R^2 of 0.813 indicates much of our data is being explained. Let's check our residuals for the validation predictions:

```python
import matplotlib.pyplot as plt
import seaborn as sns

residuals = np.abs(y_valid - dt_y_valid_predictions)


sns.scatterplot(x=dt_y_valid_predictions, y=residuals)
plt.xlabel("Instance")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot for Decision Tree Model")

threshold = 100
within_threshold = (abs(residuals) <= threshold).mean()
print(f"Proportion of residuals within {threshold}: {within_threshold:.3f}")

threshold = 50
within_threshold = (abs(residuals) <= threshold).mean()
print(f"Proportion of residuals within {threshold}: {within_threshold:.3f}")

threshold = 25
within_threshold = (abs(residuals) <= threshold).mean()
print(f"Proportion of residuals within {threshold}: {within_threshold:.3f}")
``` 

<img width="1256" height="1020" alt="image" src="https://github.com/user-attachments/assets/4817d2a2-ff36-4d53-b876-00dbf4ed15ce" /> 

Except, this plot only shows the positive side, and we can make it a bit clear with some jitter:

```python
sns.residplot(x=y_valid, y=dt_y_valid_predictions, scatter_kws={"alpha":0.5})
plt.axhline(y=0, color='r')
plt.title('Centered Residual Plot for Decision Tree Model on Validation Set')
plt.ylabel('Residual')
plt.xlabel('Instance')
``` 
<img width="1201" height="912" alt="image" src="https://github.com/user-attachments/assets/437128bd-26b6-4f10-b23e-06d1578b762e" /> 


**It's clear that a majority of our guesses were correct within 50 riders, and almost all (nearly 90%) were under 100. However, our RMSE is still pretty high, which indicates (and this is shown from the plot above) that a few points were predicted way off, probably due to some important feature like temperature that causes an outlier.**


Now apply to the test set:
```python
dt_y_test_predictions = tree_regressor.predict(X_valid)

dt_rmse_test = np.sqrt(mean_squared_error(y_valid, dt_y_test_predictions))
dt_r2_test = r2_score(y_test, dt_y_test_predictions)

mae_test = mean_absolute_error(y_test, dt_y_test_predictions)

print(f"Decision Tree - Validation MAE: {mae_test:.3f}")
print(f"Decision Tree - Validation RMSE: {dt_rmse_test:.3f}, R^2: {dt_r2_test:.3f}")
``` 
<img width="768" height="92" alt="image" src="https://github.com/user-attachments/assets/045a0442-9822-4763-91e4-364cc5cea5c4" /> 



#### Random Forest:

We'll again tune our hyperparameters:

```python
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor()

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
``` 
<img width="1917" height="43" alt="image" src="https://github.com/user-attachments/assets/02886d30-f421-4d2c-af9e-5db1d7042f57" /> 

Now add the hyperparameters to the model and apply to training and then validation sets:
```python
#Prior to Hyperparameter tuning: Random Forest - Validation RMSE: 70.366, R^2: 0.849, MAE: 46.459

#Hyperparameters:
max_depth_param = 20
max_features_param = 'sqrt'
min_samples_leaf_param = 1
min_samples_split_param = 2
n_estimators_param = 300

rf_model = RandomForestRegressor(max_depth = max_depth_param, max_features = max_features_param, min_samples_leaf = min_samples_leaf_param, min_samples_split = min_samples_split_param, n_estimators = n_estimators_param)

rf_model.fit(X_train, y_train)

y_train_forest_predictions = linear_rbf.predict(X_train)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_forest_predictions))
r2_train = r2_score(y_train, y_train_forest_predictions)

print(f"Training RMSE: {rmse_train:.3f}, R^2: {r2_train:.3f}")

rf_y_valid_predictions = rf_model.predict(X_valid)

rf_rmse_valid = np.sqrt(mean_squared_error(y_valid, rf_y_valid_predictions))
rf_r2_valid = r2_score(y_valid, rf_y_valid_predictions)
rf_mae_valid = mean_absolute_error(y_valid, rf_y_valid_predictions)

print(f"Random Forest - Validation RMSE: {rf_rmse_valid:.3f}, R^2: {rf_r2_valid:.3f}, MAE: {rf_mae_valid:.3f}")

```
Results in: 
<img width="977" height="87" alt="image" src="https://github.com/user-attachments/assets/69c9a6b4-5059-411a-9797-1ba09bfb0f9f" /> 

Now, we can do something a little interesting and check out our feature importance. I didn't even know this was an option, or how useful it really is after already tuning, but it can give us a better sense of our data. Do:

```python
importances = rf_model.feature_importances_
feature_names = X_train.columns

sorted_indices = importances.argsort()[::-1]

plt.barh(range(len(importances)), importances[sorted_indices])
plt.yticks(range(len(importances)), feature_names[sorted_indices])
plt.xlabel("Feature Importance Score")
plt.ylabel("Individual Feature")
plt.title("Random Forest Feature Importance Comparison")
```

<img width="1289" height="929" alt="image" src="https://github.com/user-attachments/assets/38e11664-f302-4427-85a7-9ef93a0f37d7" /> 

Now, I'm not certain what 'feature importance score' is really a measure of or how much we can rely on it, but from this chart it appears that hour of the day is the greatest determinant of how many riders we should expect. This probably makes sense. Rush hour exists, people go to and get off work at generally set times, so we would expect specific hours to be much busier. And since it also seems like temperature is a powerful determining factor, our cross context variable of the two seems to be helping out.

Let's apply our best-case Random Forest Model to the test set:
```python
test_rf_model = RandomForestRegressor(max_depth = max_depth_param, max_features = max_features_param, min_samples_leaf = min_samples_leaf_param, min_samples_split = min_samples_split_param, n_estimators = n_estimators_param)



test_rf_model.fit(X_train, y_train)

rf_y_test_predictions = test_rf_model.predict(X_test)

rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_y_test_predictions))
rf_r2_test = r2_score(y_test, rf_y_test_predictions)
rf_mae_test = mean_absolute_error(y_test, rf_y_test_predictions)

print(f"Random Forest - Test RMSE: {rf_rmse_test:.3f}, R^2: {rf_r2_test:.3f}, MAE: {rf_mae_test:.3f}")
``` 
<img width="872" height="57" alt="image" src="https://github.com/user-attachments/assets/acf155bb-7917-4dff-b56f-68394187b173" /> 

#### Neural Net:
Lastly, let's try out a nerual net model. Due to my limited skill with the nn tool of Torch, I doubt our model will be strong enough to supercede our decision trees and random forests, but this is a good opportunity to work on our structure building. Remember, we'll need to initialize a neural net object and introduce a forward step method to begin:

```python
import torch
from torch import nn

class Nnet(nn.Module):
    def __init__(self, input_dimension):
        super(Nnet, self).__init__()
        self.fc_1 = nn.Linear(input_dimension, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.output_layer(x)
        return x
```

And 

```python
def train_multiple_seeds(X_train, y_train, X_valid, y_valid, n_seeds=10, epochs=10):
    rmses = []
    maes = []
    train_rmses = []
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).view(-1, 1)

#train the model in each seed first
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        my_model = Nnet(X_train.shape[1])
        loss_function = nn.MSELoss()
        adam_optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            my_model.train()
            adam_optimizer.zero_grad()
            outputs = my_model(X_train_tensor)
            loss = loss_function(outputs, y_train_tensor)
            loss.backward()
            adam_optimizer.step()

        #Now evaluate on the vaidation set:
        my_model.eval()
        with torch.no_grad():
            y_train_nn_predictions = my_model(X_train_tensor)
            y_valid_predictions = my_model(X_valid_tensor)

        train_rmse = np.sqrt(mean_squared_error(y_train_tensor.numpy(), y_train_nn_predictions.numpy()))
        
        rmse = np.sqrt(mean_squared_error(y_valid_tensor.numpy(), y_valid_predictions.numpy()))
        mae_test = mean_absolute_error(y_valid_tensor.numpy(), y_valid_predictions.numpy())

        train_rmses.append(train_rmse)
        rmses.append(rmse)
        maes.append(mae_test)
        print(f"Seed {seed} - Training RMSE: {train_rmse:.3f}")
        print(f"Seed {seed} - Validation RMSE: {rmse:.3f}, MAE: {mae_test:.3f}")

    avg_train_rmse = np.mean(train_rmses)
    avg_rmse = np.mean(rmses)
    avg_mae = np.mean(maes)
    print(f"Average Training RMSE over {n_seeds} seeds: {avg_train_rmse:.3f}")
    print(f"Average Validation RMSE over {n_seeds} seeds: {avg_rmse:.3f}. Average MAE: {avg_mae:.3f}")

    return my_model, rmses, maes, train_rmses

trained_nn_model, valid_rmses, valid_maes, train_rmses = train_multiple_seeds(X_train, y_train, X_valid, y_valid, n_seeds=10)
```




### Results:
















