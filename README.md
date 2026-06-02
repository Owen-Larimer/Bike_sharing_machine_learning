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






