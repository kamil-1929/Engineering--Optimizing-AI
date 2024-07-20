import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import GridSearchCV
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

def perform_modelling_with_randomforest(X_train, Y_train):
    base_model = RandomForestClassifier(random_state=42)
    chain_model = ClassifierChain(base_model, order='random', random_state=42)
    param_grid = {
        'base_estimator__n_estimators': [50, 100, 200],
        'base_estimator__max_depth': [None, 10, 20, 30],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__min_samples_leaf': [1, 2, 4],
    }
    grid_search = GridSearchCV(chain_model, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, Y_train)
    return best_model
