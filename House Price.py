import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

# Load dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")

# Print first 5 records
print(dataset.head(5))
print("Dataset shape:", dataset.shape)

# Identify and count categorical, integer, and float variables
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Filter numeric columns for correlation
numeric_cols = dataset.select_dtypes(include=['number']).columns
plt.figure(figsize=(12, 6))
sns.heatmap(dataset[numeric_cols].corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Plot unique values in categorical features
unique_values = [dataset[col].nunique() for col in object_cols]
plt.figure(figsize=(10, 6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)
plt.show()

# Plot distribution of categorical features
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
for index, col in enumerate(object_cols, start=1):
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    sns.barplot(x=list(y.index), y=y)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Data preprocessing
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'].fillna(dataset['SalePrice'].mean(), inplace=True)

# Drop rows with missing values (except for SalePrice which is filled)
new_dataset = dataset.dropna()

# One-hot encoding
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of categorical features:', len(object_cols))

OH_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter name
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]), index=new_dataset.index)
OH_cols.columns = OH_encoder.get_feature_names_out()

df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Prepare for training
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model training and evaluation
# Support Vector Regression
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)
print("SVR Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred))

# Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
print("Random Forest Regressor Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred))

# Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
print("Linear Regression Mean Absolute Percentage Error:", mean_absolute_percentage_error(Y_valid, Y_pred))
# Print each predicted value on a new line

# CatBoost Regressor
cb_model = CatBoostRegressor(learning_rate=0.1, iterations=500, depth=10, verbose=0)  # Added parameters
cb_model.fit(X_train, Y_train)
preds = cb_model.predict(X_valid)
cb_r2_score = r2_score(Y_valid, preds)
print("CatBoost Regressor R2 Score:", cb_r2_score)
print(Y_valid)
