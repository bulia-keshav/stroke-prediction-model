# %% [markdown]
# 
# # Importing Libraries
# 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(42)

# %% [markdown]
# ### Importing dataset

# %%
data = pd.read_excel("PHD 9 Oct 2024.xlsx")

# %%
data.head()

# %%
data.shape

# %%
data.isna().sum()

# %%
data.info()

# %%
data.nunique()

# %% [markdown]
# ### Dropping unnecesaary Features

# %%
data.columns

# %%
#important_features = ['Name', 'Stroke volume', 'age', 'gender', 'NIHSS', 'SHT', 'DM', 'Alcohol', 'smoking', 'tobacco',
 #      'dyslipidaemia', 'atrial fibrillation', 'IHD',
   #    'rheumatic heart disease', 'haemoglobin', 'PCV', 'Homocystiene', 'HbA1C', 'Cholesterol', 'LDL cholesterol',
    #   'HDL Cholesterol', 'Triglycerides', 'V LDL', 'b 12', 'ecosprine',
    #   'clopidogril', 'thrombolysis', 'thrombolytic agent', 'anticoagulation',
     #  'mechanical thrombectomy', 'decompressive hemicranectomy', 'MRS',
    #   'barthel index']

# %%
important_features = ['Name', 'Stroke volume', 'age', 'gender', 'NIHSS', 'SHT', 'DM', 'Alcohol', 'smoking', 'tobacco',
      'dyslipidaemia', 'atrial fibrillation', 'IHD',
      'rheumatic heart disease', 'haemoglobin', 'Homocystiene', 'CT ASPECTS', 'MRS',
       'barthel index']

# %%
data = data[important_features]

# %%
data.head()

# %%
data.info()

# %% [markdown]
# - Storke Volume is Object , we need all the stroke volume as float

# %%
for index, value in data['Stroke volume'].items():
    # Check if each value can be converted to a float
    try:
        float(value)
    except ValueError:
        # If not a float, drop the row
        data.drop(index, inplace=True)

# %%
data.loc[:,'Stroke volume'] = pd.to_numeric(data['Stroke volume'], errors='coerce')

# %% [markdown]
# ## Dealing with NULL values

# %% [markdown]
# - Checking Null Stroke Volume

# %%
data["Stroke volume"].isna().sum()

# %% [markdown]
# - Checking NULL MRS value

# %%
data['MRS'].isna().sum()

# %%
data.shape

# %% [markdown]
# ### Dropping NULL MRS and NULL Stroke Volume
# 

# %%
data.dropna(subset=['MRS'], inplace=True)

# %%
data.shape

# %%
data['Stroke volume'].isna().sum()

# %%
data.dropna(subset=['Stroke volume'],inplace =True)

# %%
data.drop(columns=["Name"], inplace=True)

# %%
data.head()

# %%
plt.figure()
plt.title('MRS Count Plot')
sns.countplot(data=data, x='MRS')
plt.show()

# %% [markdown]
# ### Converting MRS to Good MRS and Bad MRS

# %%
# 1 = Good, 0 = Bad
data['MRS_Class'] = pd.Series()
for index, value in data['MRS'].items():
    if (value <= 2):
      data.loc[index,'MRS_Class'] = 1
    else:
      data.loc[index,'MRS_Class'] = 0

# %%
plt.figure()
plt.title('MRS Count Plot after seperating into Good and Bad')
sns.countplot(data=data, x='MRS_Class')
plt.show()

# %% [markdown]
# ## Ajusting Binary Values

# %%
binary_columns = [col for col in data.columns if set(data[col].dropna().unique()) == {1, 2}]

for col in binary_columns:
    data.loc[data[col] == 2, col] = 0

# %%
data.reset_index(inplace=True, drop=True)

# %%
data.head()

# %%
data.to_csv('Cleaned_Data.csv', index=False)
data = pd.read_csv('Cleaned_Data.csv')

# %% [markdown]
# ## Correlation Matrix

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
correlation_matrix = data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# ## Feature Extraction

# %%
y = data.iloc[:,-1].values
z = data.iloc[:,-2].values
X = data.drop(columns=["MRS","barthel index", "MRS_Class"])
X = X.to_numpy()

# %%
y

# %%
z

# %%
X

# %% [markdown]
# ## Preprocessing Pipeline

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# %%
data.isna().sum()

# %%
binary_features = [col for col in data.columns if set(data[col].dropna().unique()) == {0, 1}]
binary_features.remove('MRS_Class')
binary_features = [data.columns.get_loc(feature) for feature in binary_features]
binary_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]
)

# %%
numeric_features = ['Stroke volume', 'age', 'NIHSS', 'haemoglobin', 'Homocystiene', 'CT ASPECTS']
numeric_features = [data.columns.get_loc(feature) for feature in numeric_features]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("bin", binary_transformer, binary_features),
    ]
)

# %%
preprocessing_pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# %% [markdown]
# ## Importing Different Models

# %% [markdown]
# Top 7 Models:
# 1. LGMBClassifier
# 2. GradientBoostingClassifier
# 3. XGBoostClassifier
# 4. BaggingClassifier
# 5. AdaBoostClassifier
# 6. DecisionTreeClassifier
# 7. RandomForestClassifier
# 

# %%
# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
import warnings

# %% [markdown]
# ## Model Pipeline

# %%
pipeline = Pipeline(
    steps = [('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())]
)

# %% [markdown]
# ## Cross Validation Pipeline

# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# %%
# Define parameters for Decision Tree
param1 = {}
param1['classifier__max_depth'] = [3, 5, 10]
param1['classifier__criterion'] = ['gini', 'entropy']
param1['classifier'] = [DecisionTreeClassifier()]

# Define parameters for Random Forest
param2 = {}
param2['classifier__n_estimators'] = [50, 100, 200]
param2['classifier__max_depth'] = [3, 5, 10]
param2['classifier__criterion'] = ['gini', 'entropy']
param2['classifier__bootstrap'] = [True, False]
param2['classifier'] = [RandomForestClassifier()]

# Define parameters for AdaBoost
param3 = {}
param3['classifier__n_estimators'] = [50, 100, 200]
param3['classifier__learning_rate'] = [0.1, 0.01, 0.001]
param3['classifier__estimator'] = [DecisionTreeClassifier(criterion='gini', max_depth = 5), DecisionTreeClassifier(criterion='gini',max_depth=3)]
param3['classifier'] = [AdaBoostClassifier()]

# Define parameters for Gradient Boosting
param4 = {}
param4['classifier__n_estimators'] = [50, 100, 200]
param4['classifier__learning_rate'] = [0.1, 0.01, 0.001]
param4['classifier__max_depth'] = [3, 5, 10]
param4['classifier'] = [GradientBoostingClassifier()]

# Define parameters for XGBClassifier
param5 = {}
param5['classifier__n_estimators'] = [50, 100, 200]
param5['classifier__max_depth'] = [3, 5, 10]
param5['classifier__learning_rate'] = [0.1, 0.01, 0.001]
param5['classifier'] = [XGBClassifier()]

# Define parameters for LGBMClassifier
param6 = {}
param6['classifier__n_estimators'] = [50, 100, 200]
param6['classifier__max_depth'] = [3, 5, 10]
param6['classifier__learning_rate'] = [0.1, 0.01, 0.001]
param6['classifier'] = [LGBMClassifier()]

# Define parameters for Bagging Classifier (Example with Decision Tree)
param7 = {}
param7['classifier__n_estimators'] = [10, 50, 100]
param7['classifier__estimator'] = [DecisionTreeClassifier(criterion='gini', max_depth = 5), LogisticRegression(C = 0.1, penalty='l1', solver='liblinear')]
param7['classifier'] = [BaggingClassifier()]

# %%
params = [param1, param2, param3, param4, param5, param6, param7]

# %%
cv = KFold(n_splits=10, random_state=42, shuffle=True)

# %%
grid = GridSearchCV(pipeline, params, cv=cv, scoring=['accuracy', 'f1', 'recall', 'roc_auc', 'jaccard', 'balanced_accuracy'], refit=False, verbose=2).fit(X, y)

# %%
print(grid.best_params_)

# %%
print(grid.best_score_)

# %%
means = np.round_(grid.cv_results_['mean_test_score'], 6)
params_summary = grid.cv_results_['params']

# %%
df = pd.DataFrame(list(zip(means, params_summary)), columns=['Mean Score', 'Parmeters'])
df = df.sort_values(by = 'Mean Score', ascending=False)

# %%
from datetime import datetime
current_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
result_filename = 'Results-'+ current_datetime +'.csv'
df.to_csv(result_filename, index=False)

# %%
# CART, PERT, Random Forest Bagging & Boosting(Entropy or Gini indix), SVM(All Kernels), K_Folds

# %%
#logistic regression, Naive Bayes


