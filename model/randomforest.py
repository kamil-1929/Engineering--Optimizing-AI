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

    # Train the model
    chain_model.fit(X_train, Y_train)
    return chain_model