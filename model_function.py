# In your Library: training_code/model_train.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def run_training(df):
    print("Starting training...")

    # 1. Define the Schema
    SCHEMA = {
        'target': 'target',
        'feat_num': ['since_birth_days', 'price_first_item_purchased', 'pages_visited'],
        'feat_cat': ['ip_country_code', 'gender']
    }

    # 2. Define Transformers
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("sts", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, SCHEMA['feat_cat']),
            ("num", num_transformer, SCHEMA['feat_num'])
        ]
    )

    # 3. Define the Model Pipeline
    # Note: Changed "auto" to "sqrt" here to avoid the previous error
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    # 4. Define the Grid
    pipe_grid = {
        "model__n_estimators": [10],
        "model__max_depth": [None, 3],
        "model__max_features": ["sqrt"], 
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [5, 10]
    }

    # 5. Prepare Data
    # Ensure numerical columns are actually numbers (important for Dataiku)
    for col in SCHEMA['feat_num']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SCHEMA['feat_num'] + SCHEMA['feat_cat']]
    Y = df[SCHEMA['target']].values

    # 6. Fit
    gs_model = GridSearchCV(model, pipe_grid, n_jobs=-1, scoring='roc_auc', cv=3, verbose=2)
    gs_model.fit(X, Y)

    return gs_model.best_estimator_