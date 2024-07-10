from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .utils import string2any

class BaseModel(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self) -> int:
        """
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        return

    def build(self, values={}):
        values = values if isinstance(values, dict) else string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
