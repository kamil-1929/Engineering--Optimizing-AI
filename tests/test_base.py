import pytest
from model.base import BaseModel
from . import utils

class ConcreteModel(BaseModel):
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
