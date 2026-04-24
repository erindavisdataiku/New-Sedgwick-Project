# In your Library: training_code/model_train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def run_training(df):
    print("Starting training with passed DataFrame...")
    
    SCHEMA = {
        'target': 'target',
        'feat_num': ['since_birth_days', 'price_first_item_purchased', 'pages_visited'],
        'feat_cat': ['ip_country_code', 'gender']
    }

    # ... keep your transformer and pipeline code here ...
    
    X = df[SCHEMA['feat_num'] + SCHEMA['feat_cat']]
    Y = df[SCHEMA['target']].values
    
    gs_model = GridSearchCV(model, pipe_grid, n_jobs=-1, scoring='roc_auc', cv=3, verbose=2)
    gs_model.fit(X, Y)
    
    return gs_model.best_estimator_