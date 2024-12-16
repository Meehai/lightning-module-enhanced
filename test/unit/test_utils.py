import pytest
from typing import NamedTuple
from lightning_module_enhanced.utils import flat_if_one, make_list, tr_detach_data
import torch as tr

def test_flat_if_one_1():
    assert flat_if_one([1, 2, 3]) == [1, 2, 3]
    assert flat_if_one([[1, 2, 3]]) == [1, 2, 3]
    assert flat_if_one([1]) == 1
    assert flat_if_one([[1]]) == [1]
    with pytest.raises(AssertionError):
        _ = flat_if_one({"a": 1})
    assert flat_if_one(flat_if_one([[1, 2, 3]])) == [1, 2, 3]
    assert flat_if_one(flat_if_one([[[1, 2, 3]]])) == [1, 2, 3]
    assert flat_if_one(flat_if_one([[[1]]])) == [1]
    assert flat_if_one(flat_if_one(flat_if_one([[[1]]]))) == 1

def test_make_list_1():
    assert make_list(1) == [1]
    assert make_list([1]) == [1]
    assert make_list({"a": 1}) == [{"a": 1}]
    assert flat_if_one(make_list({"a": 1})) == {"a": 1}
    assert flat_if_one(make_list([1])) == 1
    assert flat_if_one(make_list(1)) == 1

def test_tr_detach_data():
    x = tr.randn(10, 20, requires_grad=True)
    assert tr_detach_data(x).requires_grad == False
    assert tr_detach_data({"x": x})["x"].requires_grad == False
    assert tr_detach_data((x, ))[0].requires_grad == False
    assert tr_detach_data(NamedTuple("named", x=tr.Tensor)(x=x)).x.requires_grad == False
    assert tr_detach_data([NamedTuple("named", x=tr.Tensor)(x=x), 5])[0].x.requires_grad == False
