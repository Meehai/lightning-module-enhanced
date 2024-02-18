from lightning_module_enhanced.utils import flat_if_one

def test_flat_if_one_1():
    assert flat_if_one([1, 2, 3]) == [1, 2, 3]
    assert flat_if_one([[1, 2, 3]]) == [1, 2, 3]
    assert flat_if_one([1]) == 1
    assert flat_if_one([[1]]) == [1]
    try:
        flat_if_one({"a": 1})
    except AssertionError:
        pass
    assert flat_if_one(flat_if_one([[1, 2, 3]])) == [1, 2, 3]
    assert flat_if_one(flat_if_one([[[1, 2, 3]]])) == [1, 2, 3]
    assert flat_if_one(flat_if_one([[[1]]])) == [1]
    assert flat_if_one(flat_if_one(flat_if_one([[[1]]]))) == 1
