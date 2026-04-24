

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
from xgboost import train


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# 1. Load the dataset
#print("Loading data...")
#file_path = r'/Users/erin.davis@dataiku.com/Desktop/VS Code Projects/New Sedgwick Project/clv_train_test.csv'
#df = pd.read_csv(file_path, dtype=object, nrows=50000)
#print(df.isna().sum())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Select subset of features for model training
SCHEMA = {
    'target': 'target',
    'feat_num': ['since_birth_days', 'price_first_item_purchased', 'pages_visited'],
    'feat_cat': ['ip_country_code', 'gender']
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
SCHEMA['feat_cat']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("sts", StandardScaler())
])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, SCHEMA['feat_cat']),
        ("num", num_transformer, SCHEMA['feat_num'])
    ]
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model = Pipeline(steps=[("preprocessor", preprocessor),
                       ("model", RandomForestClassifier())])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pipe_grid = {
    "model__n_estimators": [10],
    "model__max_depth": [None, 3],
    "model__max_features": ["sqrt"],  # Changed from "auto" to "sqrt"
    "model__min_samples_split": [2, 4],
    "model__min_samples_leaf": [5, 10]
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
gs_model = GridSearchCV(model, pipe_grid, n_jobs=-1, scoring='roc_auc', cv=3, verbose=2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X = df[SCHEMA['feat_num'] + SCHEMA['feat_cat']]
Y = df[SCHEMA['target']].values

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
gs_model.fit(X, Y)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
gs_model.best_params_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clf = gs_model.best_estimator_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate the final score on the training set (as a baseline)
train_roc_auc = roc_auc_score(Y, clf.predict_proba(X)[:, 1])

print(f"Model Training Complete!")
print(f"Best CV Score (ROC AUC): {gs_model.best_score_:.4f}")
print(f"Training ROC AUC: {train_roc_auc:.4f}")