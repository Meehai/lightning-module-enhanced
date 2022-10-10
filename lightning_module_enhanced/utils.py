"""Utils module"""
from typing import T, Any
import json
import torch as tr
import numpy as np
from torch import nn

from .logger import logger

def to_tensor(data: T) -> T:
    """Equivalent of the numpy's np_get_data function, converting a data structure of items to torch, where possible"""
    if data is None:
        return None

    if isinstance(data, (np.int32, np.int8, np.int16, np.int64, np.float32, np.float64, int, float)):
        return tr.Tensor([data])

    if isinstance(data, list):
        return [to_tensor(x) for x in data]

    if isinstance(data, tuple):
        return tuple(to_tensor(x) for x in data)

    if isinstance(data, set):
        return {to_tensor(x) for x in data}

    if isinstance(data, dict):
        return {k : to_tensor(data[k]) for k in data}

    if isinstance(data, tr.Tensor):
        return data

    if isinstance(data, np.ndarray):
        return tr.from_numpy(data) #pylint: disable=no-member

    if callable(data):
        return data

    if isinstance(data, str):
        return data

    logger.debug(f"Got unknown type {type(data)}")
    return data


def to_device(data: T, device: tr.device) -> T: #pylint: disable=no-member
    """Sets the torch device of the compound data"""
    if isinstance(data, (tr.Tensor, nn.Module)):
        return data.to(device)

    if isinstance(data, list):
        return [to_device(x, device) for x in data]

    if isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)

    if isinstance(data, set):
        return {to_device(x, device) for x in data}

    if isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data}

    if isinstance(data, np.ndarray):
        return tr.from_numpy(data).to(device) #pylint: disable=no-member

    if isinstance(data, (int, float, bool, str)):
        return data

    logger.debug(f"Got unknown type {type(data)}. Returning as is.")
    return data

def json_encode_val(value: Any) -> str:
    """Given a potentially unencodable json value (but stringable), convert to string if needed"""
    try:
        _ = json.dumps(value)
        encodable_value = value
    except TypeError:
        encodable_value = str(value)
    return encodable_value
