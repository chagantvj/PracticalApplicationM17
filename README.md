Project: Module 17 Practical Application Example 3
===

# Author: Vijay Chaganti

Dataset information
---

*The provided dataset contains information on 41188 customers 

*Given dataset has 41188 entries with 21 columns*

*Data Source: https://github.com/chagantvj/PracticalApplicationM17/blob/main/bank-additional-full.csv*

*Python Code: https://github.com/chagantvj/PracticalApplicationM10/blob/main/VijayChaganti_Module17_Practical_Example3.ipynb*

**Date Understanding and Cleaning**
---
*Total entries for each column of the data frame is 12684.*

```
From data given,
1. column named 'default', 'housing' and 'loan' is going to play active role on cusomers comitting to a term plan that is going to put financial strain on an individual.
2. Given priority for above three columns removing 'unknown' rows from these three columsn. Numbers of unknowns are very in-significant compared to the rows that has valid data like yes or no.
3. Column named 'job' has lot many variants that can lead over fitting of a model and hence igonored it.
4. Columns named cpi, cci, employed etc are given less proority as that data is not going to play any active role on an individual to choose to either commit for a term loan or not.

 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   age          41188 non-null  int64  
 1   job          41188 non-null  object 
 2   marital      41188 non-null  object 
 3   education    41188 non-null  object 
 4   default      41188 non-null  object 
 5   housing      41188 non-null  object 
 6   loan         41188 non-null  object 
 7   contact      41188 non-null  object 
 8   month        41188 non-null  object 
 9   day_of_week  41188 non-null  object 
 10  duration     41188 non-null  int64  
 11  campaign     41188 non-null  int64  
 12  pdays        41188 non-null  int64  
 13  previous     41188 non-null  int64  
 14  poutcome     41188 non-null  object 
 15  evr          41188 non-null  float64
 16  cpi          41188 non-null  float64
 17  cci          41188 non-null  float64
 18  e3m          41188 non-null  float64
 19  employed     41188 non-null  float64
 20  y            41188 non-null  object 
```

**Removing unknown data from columns and imputate categorical to numerical data*
---
```
countUnknown = (dfm == 'unknown').any(axis=1).sum()
print(countUnknown)
# 9359

dfm = dfm[~dfm[['loan', 'housing', 'default']].isin(['unknown']).any(axis=1)]

dfm['y'].mean()
dfm['loan'] = dfm['loan'].replace({'yes': 1, 'no': 0})
dfm['default'] = dfm['default'].replace({'yes': 1, 'no': 0})
dfm['housing'] = dfm['housing'].replace({'yes': 1, 'no': 0})

```
**Histograms of given dataset**
---
![Screen Shot 2025-01-20 at 12 58 56 PM](https://github.com/user-attachments/assets/c8de55bc-dfda-4fcc-81fd-c7dc26d7a656)

**Line graph for average calls per campaign**
---
![Screen Shot 2025-01-20 at 1 06 26 PM](https://github.com/user-attachments/assets/c1dc553b-9a25-4941-95ff-81b5af28844a)

**Code for data processing**
---
```
dfm = df.drop(['marital','job','education','month','day_of_week','pdays','previous','poutcome','cpi','cci','evr','e3m','contact'], axis=1)
dfm = dfm[~dfm[['loan', 'housing', 'default']].isin(['unknown']).any(axis=1)]
dfm['y'].mean()
dfm['loan'] = dfm['loan'].replace({'yes': 1, 'no': 0})
dfm['default'] = dfm['default'].replace({'yes': 1, 'no': 0})
dfm['housing'] = dfm['housing'].replace({'yes': 1, 'no': 0})

```
**Heatmaps withput columns 'loan', 'housing' and 'default'**
![Screen Shot 2025-01-20 at 1 08 43 PM](https://github.com/user-attachments/assets/3b74efff-fdd2-43e6-a8a8-e9064482d1fd)

**Heatmaps with columns 'loan', 'housing' and 'default'**
![Screen Shot 2025-01-20 at 1 09 04 PM](https://github.com/user-attachments/assets/9e614602-adc1-4ad4-8824-ec89d1bcf43a)
---

**Code for modeling**
```
X = dfm.drop(columns = 'y')
y = dfm['y']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)
num_columns = X_train.select_dtypes(["int","float"]).columns
num_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('num',num_transformer,num_columns)])
pipeline = Pipeline(steps = [("preprocessor",preprocessor),("classifier",LogisticRegression())])
pipeline.fit(X_train, y_train)
print(f"Train data accuracy: {pipeline.score(X_train, y_train):.2f}")
print(f"Test data accuracy: {pipeline.score(X_test, y_test):.2f}")

##Train data accuracy: 0.89
## Test data accuracy: 0.89
```

**Confusion Matrix**
---
![Screen Shot 2025-01-20 at 1 19 50 PM](https://github.com/user-attachments/assets/a940f6dd-5682-4e93-8917-caf895210975)

**Code for Model Comparison**
---
```
models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True)  # SVM with probability estimates
}

for model_name, model in models.items():
    print(f"Training {model_name}...")

    start_time = time.time()
    # Fit the model
    model.fit(X_train, y_train)
    end_time = time.time()
    runtime = end_time - start_time
    # Predict on the test set
    y_pred_test = model.predict(X_test)
    y_pred_prob_test = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    
    y_pred_train = model.predict(X_train)
    y_pred_prob_train = model.predict_proba(X_train)[:, 1]  # Probabilities for the positive class

    # Calculate evaluation metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    # Store the results
    results[model_name] = {
        'Test Accuracy': test_accuracy,
        'Train Accuracy': train_accuracy,
        'Runtime': runtime
    }

Model Comparison:

                     Test Accuracy  Train Accuracy     Runtime
KNN                       0.886706        0.916474    0.035695
Logistic Regression       0.880578        0.877696    0.119656
Decision Tree             0.858265        0.999175    0.078799
SVM                       0.877121        0.875850  102.151783
```

**Improving Model**
---
```
pipeline = Pipeline(steps = [("preprocessor",preprocessor),("classifier",KNeighborsClassifier())])
knn = KNeighborsClassifier()

param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__p': [1, 2]
}

grid_search_knn = GridSearchCV(pipeline, param_grid_knn, cv=5, scoring='accuracy', verbose=1)
grid_search_knn.fit(X_train, y_train)

print(f"Best Parameters for KNN: {grid_search_knn.best_params_}")
print(f"Best Score for KNN: {grid_search_knn.best_score_}")

# Example for Decision Tree
pipeline = Pipeline(steps = [("preprocessor",preprocessor),("classifier",DecisionTreeClassifier())])
dt = DecisionTreeClassifier()

param_grid_dt = {
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(pipeline, param_grid_dt, cv=5, scoring='accuracy', verbose=1)
grid_search_dt.fit(X_train, y_train)

print(f"Best Parameters for Decision Tree: {grid_search_dt.best_params_}")
print(f"Best Score for Decision Tree: {grid_search_dt.best_score_}")

>>> Fitting 5 folds for each of 20 candidates, totalling 100 fits
    Best Parameters for KNN:
    'classifier__n_neighbors': 11,
    'classifier__p': 2,
    'classifier__weights': 'uniform'
>>> Best Score for KNN: 0.8896003232330717


>>> Fitting 5 folds for each of 36 candidates, totalling 180 fits
    Best Parameters for Decision Tree:
       'classifier__max_depth': 10,
       'classifier__min_samples_leaf': 2,
       'classifier__min_samples_split': 2
>>> Best Score for Decision Tree: 0.8885789283372676
```


Conclusion & Recommendation
---

Out of all the columns gien in dataset, columns named 'region', 'manufacturer', 'model', 'drive', 'size', 'type', 'paint_color' & 'state' as too many categorical variables that will be overfitting the model and hence ignored those column data. If we apply One-Hot encoding on those columns, its going to generate 100s of columns which will overload the model to process.

Column named 'id', 'VIN', 'fuel', & 'cylinders' does not play a role on car price and hence dropped from the DataSet

Column named 'odometer', 'title_status' & 'condition' is going to play a role as per domain knowledge that I have and hence considered for modeling.
Applied one-hot encoding on catorigical data like 'title_status' & 'condition' and applied three different models like Linear-Regression, Losso-Regression and Ridge-Regression and all these three models gives almost ~0 on Train_R2_Score and Test_R2_Score. This shows the the models are under-fit and hence not able to predict actual price of the model

This under-fitting could be due to the 'most_frequent' imputer method used to fillin missing data for columns 'title_status' & 'condition'.

Violin plot on 'title_status' VS PriceLog & 'condition' VS PriceLog shows that the dependency is almost close to mean of the given data and hence the model. This data suggests that the model might be able to predict actual value given the data considered for models designed.

Overall the suggestions
