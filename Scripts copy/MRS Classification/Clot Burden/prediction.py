# %% [markdown]
# 
# ## Importing Libraries
# 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(42)

# %% [markdown]
# ## Importing dataset

# %%
file_path = 'Cleaned Sheet.xlsx'

data = pd.read_excel(file_path)

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
# ### Selecting features manually

# %%
data.columns

# %%
# important_features = ['Name', 'COMPLETE', 'Stroke volume', 'age', 'gender', 'NIHSS', 'SHT',
#        'DM', 'Alcohol', 'tobacco', 'smoking', 'dyslipidaemia',
#        'atrial fibrillation', 'IHD', 'rheumatic heart disease',
#        'past history of stroke/TIA', 'haemoglobin', 'PCV', 'MCV',
#        'Homocystiene', 'HbA1C', 'Cholesterol', 'LDL Cholesterol',
#        'HDL Cholesterol', 'Triglycerides', 'V LDL', 'b 12', 'Vit D',
#        'CT ASPECTS', 'TAN', 'MAS', 'MITEFF', 'MCTA', 'collaterals',
#        'ecosprine', 'clopidogril', 'thrombolysis', 'thrombolytic agent',
#        'anticoagulation', 'mechanical thrombectomy',
#        'decompressive hemicranectomy', 'MRS', 'barthel index',
#        'Rt infraclinoid ICA', 'Rt Supraclinoid ICA', 'Rt Proximal M1 MCA',
#        'Rt Distal M1 MCA', 'Rt M2MCA rear', 'Rt M2 MCA forward', 'Rt A1 ACA',
#        'Lt infraclinoid ICA', 'Lt Supraclinoid ICA', 'Lt Proximal M1 MCA',
#        'Lt Distal M1 MCA', 'Lt M2MCA rear', 'Lt M2 MCA forward', 'Lt A1 ACA',
#        'clot burden score', 'Lt ICA origin', 'Rt ICA origin', 'CCA']

# %%
important_features = ['Name','Stroke volume', 'age', 'gender', 'NIHSS', 'SHT','DM', 'Alcohol', 
                      'tobacco', 'smoking', 'dyslipidaemia', 'atrial fibrillation', 'IHD',
                      'rheumatic heart disease', 'haemoglobin', 'PCV', 
                      'MCV', 'Homocystiene', 'HbA1C', 'Cholesterol', 'HDL Cholesterol', 
                      'Triglycerides', 'V LDL', 'b 12', 'collaterals', 'mechanical thrombectomy', 
                      'decompressive hemicranectomy', 'MRS','clot burden score']

# %%
# Remove 'MRS' and 'barthel index' from their current positions
important_features.remove('MRS')

# Append 'MRS' and 'barthel index' to the end
important_features.extend(['MRS'])

# Reorder the DataFrame columns
data = data[important_features]

# %%
data.head()

# %%
data.info()

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# ## Dealing with NULL values

# %%
# Replace missing values with 0 and also convert 2.0 to 0 and 1.0 to 1
columns_to_modify = ['mechanical thrombectomy', 'decompressive hemicranectomy']

# Replace missing values with 0
data[columns_to_modify] = data[columns_to_modify].fillna(0)

# Replace 2.0 with 0 and 1.0 remains as 1
data[columns_to_modify] = data[columns_to_modify].replace(2.0, 0).replace(1.0, 1)

# %%
# Calculate the percentage of missing values for each column
missing_percentage = (data.isna().sum() / len(data)) * 100

# Filter only the columns that have missing values > 20%
missing_percentage = missing_percentage[missing_percentage > 20]

print(missing_percentage)

# %% [markdown]
# ### Dropping columns with >20% missing values
# 

# %%
# Identify columns with more than 20% missing values
columns_to_drop = missing_percentage[missing_percentage > 20].index

# Drop these columns
data = data.drop(columns=columns_to_drop)

# %%
# Display the remaining columns
print("Dropped columns:", columns_to_drop)
print("Remaining columns:", data.columns)

# %%
data.shape

# %% [markdown]
# ### Drop Name

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
plt.title('MRS Count Plot after separating into Good and Bad')

# Set the x variable to hue and use integer keys for the palette
sns.countplot(data=data, x='MRS_Class', hue='MRS_Class', palette={0: 'green', 1: 'red'}, legend=False)

# Set the custom x-tick labels
plt.xticks(ticks=[0, 1], labels=['Good', 'Bad'])

plt.show()

# %% [markdown]
# ## Ajusting Columns

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
X = data.drop(columns=["MRS", "MRS_Class"])
X = X.to_numpy()

# %%
y

# %%
X

# %%
X.shape

# %% [markdown]
# ## Preprocessing Pipeline

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# %%
data.isna().sum()

# %%
# Identify binary features
binary_features = [col for col in data.columns if set(data[col].dropna().unique()) == {0, 1}]
binary_features.remove('MRS_Class')

# Get indices of binary features in the DataFrame
binary_features_indices = [data.columns.get_loc(feature) for feature in binary_features]

# Create the binary transformer
binary_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]
)

# %%
data.columns[binary_features_indices]

# %%
# Identify numeric features by excluding the binary features from the DataFrame
numeric_features = [col for col in data.columns if col not in binary_features]
# Remove both 'MRS_Class' and 'MRS' in a single line
numeric_features = [feature for feature in numeric_features if feature not in ['MRS_Class', 'MRS']]

# Get indices of numeric features in the DataFrame
numeric_features_indices = [data.columns.get_loc(feature) for feature in numeric_features]

# Create the numeric transformer
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
)

# %%
data.columns[numeric_features_indices]

# %%
# Create the column transformer for the entire preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features_indices),
        ("bin", binary_transformer, binary_features_indices),
    ]
)

# %%
preprocessing_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# %% [markdown]
# # Prediction

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
    steps=[
        ('preprocessor', preprocessing_pipeline),
        ('classifier', DecisionTreeClassifier())
    ]
)

# %% [markdown]
# ## Cross Validation Pipeline

# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# %%
# Define parameters for GridSearchCV
param_grid = [
    # Parameters for Decision Tree
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [5, 10, None],
        'classifier__criterion': ['gini', 'entropy']
    },
    # Parameters for Random Forest
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 20, 30],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__bootstrap': [True]
    },
    # Parameters for AdaBoost
    {
        'classifier': [AdaBoostClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.1, 0.01],
        'classifier__estimator': [DecisionTreeClassifier(max_depth=3)]
    },
    # Parameters for Gradient Boosting
    {
        'classifier': [GradientBoostingClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.1, 0.01],
        'classifier__max_depth': [3, 5]
    },
    # Parameters for XGBoost
    {
        'classifier': [XGBClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.01]
    },
    # Parameters for LGBM
    {
        'classifier': [LGBMClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 10],
        'classifier__learning_rate': [0.1, 0.01],
        'classifier__num_leaves': [7, 15, 31]
    },
    # Parameters for Bagging Classifier
    {
        'classifier': [BaggingClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__estimator': [DecisionTreeClassifier(max_depth=5)]
    }
]

# %% [markdown]
# ## Grid Search Cross Validation

# %%
cv = KFold(n_splits=10, random_state=42, shuffle=True)

# %%
# grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=['accuracy', 'f1', 'recall', 'roc_auc', 'jaccard', 'balanced_accuracy'], refit=False, verbose=2)
# grid.fit(X, y)

# %%
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=['accuracy', 'f1', 'recall', 'roc_auc', 'jaccard', 'balanced_accuracy'], refit='f1', verbose=2)
grid.fit(X, y)

# %% [markdown]
# # Saving Model

# %%
import joblib

# Save the entire GridSearchCV object
grid_filename = 'GSCV_NOFS_CBS_Col.pkl'
joblib.dump(grid, grid_filename)

print(f"Complete GridSearchCV model saved as '{grid_filename}'")

# %%
print(grid.best_params_)

# %%
print(grid.best_score_)

# %%
from datetime import datetime

# List of scoring metrics used in GridSearchCV
scoring_metrics = ['accuracy', 'f1', 'recall', 'roc_auc', 'jaccard', 'balanced_accuracy']

# Extracting the parameter settings for each run and combining them into one column
params_summary = grid.cv_results_['params']
combined_params = [str(param_set) for param_set in params_summary]

# Creating an initial DataFrame with combined parameters
df = pd.DataFrame({'Parameters': combined_params})

# Adding mean scores for each scoring metric to the DataFrame
for metric in scoring_metrics:
    mean_score_key = f'mean_test_{metric}'
    
    if mean_score_key in grid.cv_results_:
        # Extract the mean scores and add to the DataFrame
        df[f'Mean {metric.capitalize()} Score'] = np.round_(grid.cv_results_[mean_score_key], 6)
    else:
        print(f"Metric '{metric}' not found in cv_results_")

# Sorting by F1 Score as it was the refit metric
df = df.sort_values(by='Mean F1 Score', ascending=False)

# Saving the DataFrame to an Excel file
current_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
result_filename = 'MRS Prediction Results CBS and Collaterals with NO FS--' + current_datetime + '.xlsx'
df.to_excel(result_filename, index=False)

print(f"Results saved to {result_filename}")


# %% [markdown]
# # Confusion Matrix + Model Evaluation

# %% [markdown]
# ## Load Libraries

# %%
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    f1_score, recall_score, roc_auc_score, jaccard_score, 
    balanced_accuracy_score, precision_score
)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold

# %% [markdown]
# ## Load Best GSCV Model

# %%
# Load the saved GridSearchCV object
grid_filename = 'GSCV_NOFS_CBS_Col.pkl'
grid = joblib.load(grid_filename)

# Extract the best model from the loaded GridSearchCV object
best_model = grid.best_estimator_

print("Best Model Loaded Successfully!")

# %% [markdown]
# ## Load and Preprocess the Data

# %%
# Load your cleaned data
data = pd.read_csv('Cleaned_Data.csv')

# Separate features and target variable
y = data.iloc[:, -1].values
X = data.drop(columns=["MRS", "MRS_Class"]).to_numpy()

# # Handle missing values by imputing them
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# %% [markdown]
# ## Perform K-Fold CV

# %%
# Perform K-Fold cross-validation with 10 splits
kf = KFold(n_splits=10, random_state=42, shuffle=True)

# Use cross_val_predict to get cross-validated predictions
y_pred = cross_val_predict(best_model, X, y, cv=kf)

print("Cross-Validation Completed!")

# %% [markdown]
# ## Confusion Matrix

# %%
# Generate confusion matrix for cross-validated predictions
conf_matrix = confusion_matrix(y, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Cross-Validated Predictions')

# Save the plot
plt.savefig('Confusion_Matrix_Best_Performing_GSCV_NOFS_MRS_PRED_CBS_COLLATERAL.png', dpi=300)
plt.show()

print("Confusion Matrix Generated and Saved!")

# %% [markdown]
# ## Classification Report

# %%
# Generate and print a detailed classification report for cross-validated predictions
classification_rep = classification_report(y, y_pred)
print("\nClassification Report for Cross-Validated Predictions:\n", classification_rep)

# %% [markdown]
# ## CV Scores

# %%
# Perform cross-validation on the best model to assess overfitting
# Cross-validate using 10 folds and the 'f1' scoring metric
cross_val_scores = cross_val_score(best_model, X, y, cv=kf, scoring='f1')

# Print scores
print("\nCross-Validation F1 Scores:", cross_val_scores)
print("Mean F1 Score from Cross-Validation:", np.mean(cross_val_scores))

# %% [markdown]
# ## Extract Best Performing Models

# %%
# Extract GridSearchCV results into DataFrame
df_results = pd.DataFrame(grid.cv_results_)
df_results = df_results.sort_values(by='mean_test_f1', ascending=False)

# Get unique top-performing model instances
unique_models = df_results['param_classifier'].unique()
top_models = []

for model in unique_models:
    top_model = df_results[df_results['param_classifier'] == model].head(1)
    top_models.append(top_model)

# Concatenate all top-performing unique models
top_models_df = pd.concat(top_models)

print("Top-Performing Models Extracted!")

# %% [markdown]
# ## Metrics of Top Performing Models

# %%
# Initialize a list to store metrics for each unique top model
metrics_list = []

# Evaluate metrics for the top-performing unique models using cross-validation
for index, row in top_models_df.iterrows():
    model_params = row['params']
    
    # Set the model with the corresponding parameters
    model = model_params['classifier']
    model.set_params(**{
        key.replace('classifier__', ''): value 
        for key, value in model_params.items() 
        if key.startswith('classifier__')
    })
    
    # Perform cross-validation and predict
    y_pred = cross_val_predict(model, X, y, cv=kf)
    
    # Calculate metrics
    metrics = {
        'Model': type(model).__name__,
        'Accuracy': accuracy_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'ROC AUC': roc_auc_score(y, y_pred),
        'Jaccard': jaccard_score(y, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y, y_pred)
    }
    metrics_list.append(metrics)

print("Performance Metrics Computed!")

# %% [markdown]
# ## Save Metrics

# %%
# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Save metrics to an Excel file
metrics_filename = 'Top_Unique_Models_Metrics_MRS_PRED_CBS_COLL_NOFS.xlsx'
metrics_df.to_excel(metrics_filename, index=False)

print(f"Metrics saved to '{metrics_filename}'!")


