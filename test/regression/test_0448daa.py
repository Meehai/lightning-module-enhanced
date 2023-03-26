from lightning_module_enhanced import LME
from torch import nn
from copy import deepcopy
import torch as tr

def test_0448daa():
    """
    Regression test.
    https://gitlab.com/mihaicristianpirvu/lightning-module-enhanced/-/commit/0448daa18cc4414dc7377c3dfa8cb58b0e83e747
    This tests that if we have parameters(), but no reset_parameters(), then we'll try to recursively call
    reset_parameters(), by first converting the model to a LME.
    """
    module = LME(nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))))
    params = deepcopy(tuple(module.parameters()))
    module.reset_parameters()
    new_params = deepcopy(tuple(module.parameters()))
    for p1, p2 in zip(params, new_params):
        assert not tr.allclose(p1, p2)
