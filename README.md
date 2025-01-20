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

**Histogram plot of log of price**
---
<img width="653" alt="Screenshot 2024-11-17 at 10 56 37 PM" src="https://github.com/user-attachments/assets/4339f799-d9b3-4659-b8f8-96a1643c7f0e">

**Lost data with Z-SCORE (< 1%) vs IRQ (98%)**
---
```
zscore_data_lost = 1 - (df_zscore.shape[0]/df.shape[0])
print("We lost {:.6%} of the data by the z-score method" .format(zscore_data_lost))
# We lost 0.007262% of the data by the z-score method <---- We lost only less than 1% of data using zscore
df_zscore['price'].describe()
count    426849.00
mean      17552.14
std       20667.53
min           0.00
25%        5900.00
50%       13950.00
75%       26455.00
max     5000000.00

irq_data_lost = 1 - (df_irq.shape[0]/df.shape[0])
print("We lost {:.2%} of the data by the IRQ method" .format(irq_data_lost))
# We lost 98.08% of the data by the IRQ method <--- We lost almost 98% of data with IRQ method of eliminating data which is not good.
df_irq['price'].describe()
count         8177.00
mean       3088930.26
std       87973256.90
min          57400.00
25%          61000.00
50%          67995.00
75%          77999.00
max     3736928711.00 
```
**Violin plot of title_status vs price log**
---
<img width="716" alt="Screenshot 2024-11-17 at 11 07 29 PM" src="https://github.com/user-attachments/assets/ddf85b84-473f-4f66-9387-f1788c5b27ef">

**Violin plot of condition vs price log**
---
<img width="654" alt="Screenshot 2024-11-17 at 11 07 57 PM" src="https://github.com/user-attachments/assets/abf1c186-cb5d-4945-a6e3-f51581c2e25f">


**Train & Test data with 20% Test Size and random state as 23**
---
**Applied One-Hot Encoding on Nominal category of columns like title_status & condition**
---
```
dfc = df[['condition', 'title_status', 'odometer’]]
X = dfc[['condition', 'title_status', 'odometer']]
y = df_numeric['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state=23)

from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy = "most_frequent")
X_train = si.fit_transform(X_train)
X_test = si.transform(X_test)

pd.get_dummies(dfc.select_dtypes("object"), dtype = int, drop_first = True)
# 426880 rows × 10 columns

y.mean().round(1)
# 75199.0 
```

**Linear Regression**
---
<img width="1109" alt="Screenshot 2024-11-18 at 7 10 53 PM" src="https://github.com/user-attachments/assets/7e249944-8fbd-4567-915c-7f62d18cbfd9">

**Losso Regression**
---
<img width="1137" alt="Screenshot 2024-11-18 at 10 36 15 PM" src="https://github.com/user-attachments/assets/c7fda4ee-c558-421c-9372-3ba8bf133f18">

**Ridge Regression**
---
<img width="1135" alt="Screenshot 2024-11-18 at 10 37 23 PM" src="https://github.com/user-attachments/assets/53bb46a5-edd7-4f10-b5bc-30fe40a19b86">

Conclusion & Recommendation
---

Out of all the columns gien in dataset, columns named 'region', 'manufacturer', 'model', 'drive', 'size', 'type', 'paint_color' & 'state' as too many categorical variables that will be overfitting the model and hence ignored those column data. If we apply One-Hot encoding on those columns, its going to generate 100s of columns which will overload the model to process.

Column named 'id', 'VIN', 'fuel', & 'cylinders' does not play a role on car price and hence dropped from the DataSet

Column named 'odometer', 'title_status' & 'condition' is going to play a role as per domain knowledge that I have and hence considered for modeling.
Applied one-hot encoding on catorigical data like 'title_status' & 'condition' and applied three different models like Linear-Regression, Losso-Regression and Ridge-Regression and all these three models gives almost ~0 on Train_R2_Score and Test_R2_Score. This shows the the models are under-fit and hence not able to predict actual price of the model

This under-fitting could be due to the 'most_frequent' imputer method used to fillin missing data for columns 'title_status' & 'condition'.

Violin plot on 'title_status' VS PriceLog & 'condition' VS PriceLog shows that the dependency is almost close to mean of the given data and hence the model. This data suggests that the model might be able to predict actual value given the data considered for models designed.

Overall the suggestions
