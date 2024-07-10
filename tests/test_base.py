import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from model.base import BaseModel

class ConcreteModel(BaseModel):
    def __init__(self):
        self.defaults = {}
        super().__init__()

    def train(self):
        pass

    def predict(self):
        return 1

    def data_transform(self):
        return

def test_concrete_model_initialization():
    model = ConcreteModel()
    assert isinstance(model, BaseModel)

def test_build_method_with_dict():
    model = ConcreteModel()
    values = {'key1': 'value1', 'key2': 'value2'}
    model.build(values)
    assert model.key1 == 'value1'
    assert model.key2 == 'value2'

def test_build_method_with_string():
    model = ConcreteModel()
    values = "key1:value1,key2:value2"
    model.build(values)
    assert model.key1 == 'value1'
    assert model.key2 == 'value2'
