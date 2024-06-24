import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import optuna
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tabulate import tabulate
from sklearn import metrics

# Ignore warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Settings
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Load Data
train = pd.read_csv("C:/Users/helfo/OneDrive/Documents/GitHub/Network intrusion/Dataset/Train_data.csv")
test = pd.read_csv("C:/Users/helfo/OneDrive/Documents/GitHub/Network intrusion/Dataset/Test_data.csv")

# Initial Data Inspection
print(f"Training data has {train.shape[0]} rows & {train.shape[1]} columns")
print(f"Testing data has {test.shape[0]} rows & {test.shape[1]} columns")
print(train.head(4))
print(test.head(4))
print(train.info())
print(train.describe())
print(train.describe(include='object'))

# Check for Missing Data
total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")

# Check for Duplicates
print(f"Number of duplicate rows: {train.duplicated().sum()}")

# Check for Outliers
# Use boxplots and scatter plots if necessary (omitted here for brevity)

# Feature Correlation
plt.figure(figsize=(40,30))
sns.heatmap(train.corr(), annot=True)

# Class Distribution
sns.countplot(x=train['class'])

# Label Encoding
def label_encode(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

label_encode(train)
label_encode(test)

# Drop redundant column
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Feature Selection
X_train = train.drop(['class'], axis=1)
Y_train = train['class']

rfc = RandomForestClassifier()
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)
selected_features = X_train.columns[rfe.support_]

X_train = X_train[selected_features]

# Split and scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test = scaler.fit_transform(test)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

# Model Definitions and Training
def train_and_evaluate_model(model, model_name):
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"{model_name} - Train Score: {train_score}, Test Score: {test_score}")
    return train_score, test_score

# K Nearest Neighbors (KNN) classification model with Optuna
def knn_objective(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16, log=False)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

study_KNN = optuna.create_study(direction='maximize')
study_KNN.optimize(knn_objective, n_trials=50)
best_knn = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors'])
KNN_train, KNN_test = train_and_evaluate_model(best_knn, "KNN")

# Logistic Regression Model
lg_model = LogisticRegression(random_state=42)
lg_train, lg_test = train_and_evaluate_model(lg_model, "Logistic Regression")

# Decision Tree Classifier with Optuna
def dt_objective(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32, log=False)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10, log=False)
    model = DecisionTreeClassifier(max_features=dt_max_features, max_depth=dt_max_depth)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(dt_objective, n_trials=50)
best_dt = DecisionTreeClassifier(max_features=study_dt.best_trial.params['dt_max_features'], max_depth=study_dt.best_trial.params['dt_max_depth'])
dt_train, dt_test = train_and_evaluate_model(best_dt, "Decision Tree")

# Random Forest Classifier with Optuna
def rf_objective(trial):
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=False)
    rf_max_features = trial.suggest_int('rf_max_features', 2, 10, log=False)
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 3, 100, log=False)
    model = RandomForestClassifier(max_features=rf_max_features, max_depth=rf_max_depth, n_estimators=rf_n_estimators)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(rf_objective, n_trials=50)
best_rf = RandomForestClassifier(max_features=study_rf.best_trial.params['rf_max_features'], max_depth=study_rf.best_trial.params['rf_max_depth'], n_estimators=study_rf.best_trial.params['rf_n_estimators'])
rf_train, rf_test = train_and_evaluate_model(best_rf, "Random Forest")

# Gradient Boosting Model
SKGB = GradientBoostingClassifier(random_state=42)
SKGB_train, SKGB_test = train_and_evaluate_model(SKGB, "Gradient Boosting")

# XGBoost Model
xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)
xgb_train, xgb_test = train_and_evaluate_model(xgb_model, "XGBoost")

# LightGBM Model
lgb_model = LGBMClassifier(random_state=42)
lgb_train, lgb_test = train_and_evaluate_model(lgb_model, "LightGBM")

# AdaBoost Model
ab_model = AdaBoostClassifier(random_state=42)
ab_train, ab_test = train_and_evaluate_model(ab_model, "AdaBoost")

# CatBoost Model
cb_model = CatBoostClassifier(verbose=0)
cb_train, cb_test = train_and_evaluate_model(cb_model, "CatBoost")

# Naive Bayes Model
BNB_model = BernoulliNB()
BNB_train, BNB_test = train_and_evaluate_model(BNB_model, "Naive Bayes")

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('KNN', best_knn), 
    ('Logistic Regression', lg_model), 
    ('Decision Tree', best_dt), 
    ('Random Forest', best_rf), 
    ('Gradient Boosting', SKGB), 
    ('XGBoost', xgb_model), 
    ('LightGBM', lgb_model), 
    ('AdaBoost', ab_model), 
    ('CatBoost', cb_model)], 
    voting='hard')
voting_train, voting_test = train_and_evaluate_model(voting_clf, "Voting Classifier")

# Support Vector Machine (SVM) Model with Optuna
def svm_objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'linearSVC'])
    c = trial.suggest_float('C', 0.02, 1.0, step=0.02)
    if kernel in ['linear', 'rbf']:
        model = SVC(kernel=kernel, C=c)
    elif kernel == 'linearSVC':
        model = LinearSVC(C=c)
    elif kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 10)
        model = SVC(kernel=kernel, C=c, degree=degree)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(svm_objective, n_trials=50)
kernel = study_svm.best_trial.params['kernel']
c = study_svm.best_trial.params['C']
if kernel in ['linear', 'rbf']:
    svm_model = SVC(kernel=kernel, C=c)
elif kernel == 'linearSVC':
    svm_model = LinearSVC(C=c)
elif kernel == 'poly':
    degree = study_svm.best_trial.params['degree']
    svm_model = SVC(kernel=kernel, C=c, degree=degree)
svm_train, svm_test = train_and_evaluate_model(svm_model, "SVM")

# Summary of Results
models_summary = {
    "Model": ["KNN", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "AdaBoost", "CatBoost", "Naive Bayes", "Voting Classifier", "SVM"],
    "Train Accuracy": [KNN_train, lg_train, dt_train, rf_train, SKGB_train, xgb_train, lgb_train, ab_train, cb_train, BNB_train, voting_train, svm_train],
    "Test Accuracy": [KNN_test, lg_test, dt_test, rf_test, SKGB_test, xgb_test, lgb_test, ab_test, cb_test, BNB_test, voting_test, svm_test]
}
results_df = pd.DataFrame(models_summary)
print(tabulate(results_df, headers='keys', tablefmt='pretty'))

# Confusion Matrix for the best model
best_model = xgb_model
y_pred = best_model.predict(x_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - XGBoost')
plt.show()
