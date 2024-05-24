import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings

warnings.simplefilter(action='ignore', category=Warning)

# Load datasets
attributes = pd.read_csv('/kaggle/input/scotium/scoutium_attributes.csv', sep=';')
potential_labels = pd.read_csv('/kaggle/input/scotium/scoutium_potential_labels.csv', sep=';')

# Merge two datasets
df = attributes.merge(potential_labels, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'])

# Remove the goalkeeper position from the data
df = df[~(df.position_id == 1)]

# Remove the below_average class from data because the percentage of this class is so small
df = df[~(df.potential_label == 'below_average')]

# Create pivot table
df_final = df.pivot_table(values='attribute_value', index=['player_id', 'position_id', 'potential_label'],
                          columns=['attribute_id']).reset_index()
df_final.columns = df_final.columns.astype(str)

# Encode potential_label
label_encoder = LabelEncoder()
df_final.potential_label = label_encoder.fit_transform(df_final.potential_label)
list(label_encoder.classes_)

# Create a new feature from assessments
df_final['average_value'] = df_final.iloc[:, 3:].apply(lambda x: (x.sum() / len(x)), axis=1)

# Standardize the numerical columns
num_cols = [col for col in df_final.columns if col not in ['player_id', 'position_id', 'potential_label']]
standard_scaler = StandardScaler()
df_final[num_cols] = standard_scaler.fit_transform(df_final[num_cols])


# Function to analyze the target variable
def target_analysis(dataframe, target_name):
    # Display the count and ratio of each target class
    print(pd.DataFrame({target_name: dataframe[target_name].value_counts(),
                        'Ratio': 100 * dataframe[target_name].value_counts() / len(dataframe)}))
    print('#' * 50)

    # Plot the distribution of the target variable
    plt.figure(figsize=(9, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(dataframe[target_name])
    plt.title('Histogram')
    plt.xlabel('Potential Label')
    plt.subplot(1, 2, 2)
    plt.pie(dataframe[target_name].value_counts(), labels=['Average', 'Highlighted'], autopct='%1.1f%%')
    plt.title('Pie Chart')
    plt.xlabel('Potential Label')
    plt.show()


# Analyze the target variable
target_analysis(df_final, 'potential_label')

# Display the first few rows of the dataframe
df_final.head()


# Function to validate models
def model_validation(dataframe, target, plot=False, random=17):
    accuracy_list = []
    f1_list = []
    roc_auc_list = []

    # Data Preparation
    num_cols = [col for col in dataframe.columns if col not in ['player_id', 'position_id', 'potential_label']]
    X = dataframe[num_cols]
    y = dataframe[target]

    # Model list
    model_list = {'Logistic Regression': LogisticRegression(random_state=random),
                  'KNN': KNeighborsClassifier(),
                  'CART': DecisionTreeClassifier(random_state=random),
                  'Random Forests': RandomForestClassifier(random_state=random),
                  'GBM': GradientBoostingClassifier(random_state=random),
                  'XGBoost': XGBClassifier(random_state=random),
                  'CatBoost': CatBoostClassifier(random_state=random, verbose=False)}

    # Cross validation for each model
    for idx, model in model_list.items():
        print(f'------------------{idx}------------------')
        cv_results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        for score in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']:
            print(f"{score.replace('test_', '')}: {round(cv_results[score].mean(), 4)}")
            if score == 'test_accuracy':
                accuracy_list.append(round(cv_results[score].mean(), 4))
            elif score == 'test_f1':
                f1_list.append(round(cv_results[score].mean(), 4))
            elif score == 'test_roc_auc':
                roc_auc_list.append(round(cv_results[score].mean(), 4))
        print(f'-----------------------------------------')

    # Plot the results
    if plot:
        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        sns.barplot(x=accuracy_list, y=list(model_list.keys()))
        plt.title('Accuracy Score')
        plt.subplot(3, 1, 2)
        sns.barplot(x=f1_list, y=list(model_list.keys()))
        plt.title('F1 Score')
        plt.subplot(3, 1, 3)
        sns.barplot(x=roc_auc_list, y=list(model_list.keys()))
        plt.title('ROC AUC Score')
        plt.show()


# Validate models
model_validation(df_final, 'potential_label', plot=True)


# Function for hyperparameter tuning
def hyperparameter_tuning(dataframe, target, plot=False, random=17):
    accuracy_list = []
    f1_list = []
    roc_auc_list = []
    best_params = {}

    # Data Preparation
    num_cols = [col for col in dataframe.columns if col not in ['player_id', 'position_id', 'potential_label']]
    X = dataframe[num_cols]
    y = dataframe[target]

    # Hyperparameter Sets
    knn_params = {'n_neighbors': range(2, 11)}
    cart_params = {'max_depth': range(1, 11), 'min_samples_split': range(2, 20)}
    rf_params = {"max_depth": [5, 8, 12, None], "max_features": [3, 5, 7, 15, "sqrt"],
                 "min_samples_split": [2, 5, 8, 15, 20], "n_estimators": [100, 200, 500]}
    gbm_params = {'learning_rate': [0.1, 0.01], 'max_depth': [3, 8, 10, 15], 'n_estimators': [100, 500, 1000, 2000],
                  'subsample': [1, 0.5, 0.7]}
    xgboost_params = {'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [5, 8, 12, 15, 20, None],
                      'n_estimators': [100, 500, 1000, 2000], 'colsample_bytree': [0.5, 0.7, 1]}
    catboost_params = {'iterations': [200, 500, 1000], 'learning_rate': [0.1, 0.01, 0.001], 'depth': [3, 6]}

    hyperparameter_sets = {'KNN': knn_params, 'CART': cart_params, 'Random Forests': rf_params, 'GBM': gbm_params,
                           'XGBoost': xgboost_params, 'CatBoost': catboost_params}
    model_list = {'KNN': KNeighborsClassifier(), 'CART': DecisionTreeClassifier(random_state=random),
                  'Random Forests': RandomForestClassifier(random_state=random),
                  'GBM': GradientBoostingClassifier(random_state=random), 'XGBoost': XGBClassifier(random_state=random),
                  'CatBoost': CatBoostClassifier(random_state=random, verbose=False)}

    # GridSearch Process for hyperparameter tuning
    for idx, model in model_list.items():
        best_model_grid = GridSearchCV(model, hyperparameter_sets[idx], cv=5, n_jobs=-1, verbose=True).fit(X, y)
        print(f'------------------{idx}------------------')
        cv_results = cross_validate(model.set_params(**best_model_grid.best_params_), X, y, cv=5,
                                    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        print(f'-----------------------------------------')
        best_params[idx] = best_model_grid.best_params_
        for score in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']:
            print(f"{score.replace('test_', '')}: {round(cv_results[score].mean(), 4)}")
            if score == 'test_accuracy':
                accuracy_list.append(round(cv_results[score].mean(), 4))
            elif score == 'test_f1':
                f1_list.append(round(cv_results[score].mean(), 4))
            elif score == 'test_roc_auc':
                roc_auc_list.append(round(cv_results[score].mean(), 4))

    # Plot the results
    if plot:
        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        sns.barplot(x=
