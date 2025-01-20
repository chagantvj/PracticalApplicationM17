Project: Module 17 Practical Application Example 3
===

# Author: Vijay Chaganti

Dataset information
---

*The provided dataset contains information on 426K cars to ensure speed of processing. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.*

*Given dataset has 426880 entries with 18 columns*

*Data Source: https://github.com/chagantvj/PracticalApplicationM10/blob/main/vehicles.csv.zip*

*Python Code: https://github.com/chagantvj/PracticalApplicationM10/blob/main/VijayChaganti_Module17_Practical_Example3.ipynb*

*Note: There are no errors in the code, but FutureWarning due to usage of replace=inplace option.*

**Date Understanding and Cleaning**
---
*Total entries for each column of the data frame is 12684.*

*There is some data missing for Columsn car, Bar, CoffeeHouse, CarryAway, RestaurantLessThan20 and Restaurant20To50.*

```
Many columns have missing data up to 40+ pergentage and
many columns are irrelevant to the price of car such as VIN, drive, paint_color, state, id etc

df.isna().mean().round(4)*100
id               0.00
region           0.00
price            0.00
year             0.28
manufacturer     4.13
model            1.24
condition       40.79
cylinders       41.62
fuel             0.71
odometer         1.03
title_status     1.93
transmission     0.60
VIN             37.73
drive           30.59
size            71.77
type            21.75
paint_color     30.50
state            0.00
dtype: float64

p = [0.05, 0.85, 0.95, 0.99]
df[['price']].describe(p)
price
count	426880.00
mean	75199.03
std	12182282.17
min	0.00
5%	0.00
50%	13950.00
85%	32995.00
95%	44500.00
99%	66995.00
max	3736928711.00
```

**Replaced NaN and INF values in price column with mean of price!* and no duplicate data found*
---
```
has_nan = df['price'].isna().any().sum() 
has_inf = np.isinf(df['price']).any().sum()
df_cleaned = df[~np.isinf(df['price'])]
has_inf = np.isinf(df_cleaned['price']).any().sum()
print(has_nan)
#  0
print(has_inf)
#  0

sum(df.duplicated()) ## O duplicates found
#  0
```
**Boxplot of price log**
---
<img width="654" alt="Screenshot 2024-11-17 at 10 54 37 PM" src="https://github.com/user-attachments/assets/b9b2ec9b-02c8-4593-8533-e4f5fb4c5b6f">

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
