import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random

num_folds = 0
seed = 0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class GradientBoosting(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(GradientBoosting, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = GradientBoostingClassifier(n_estimators=1000, random_state=seed)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions
        return predictions

    def print_results(self, data):
        print(classification_report(data.y_test, self.predictions, zero_division=0))

    def data_transform(self):
        # Implement data transformation logic here if needed
        pass
