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


#### Decision Trees:


#### Random Forest:


#### Neural Net:


### Results:
















