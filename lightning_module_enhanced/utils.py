"""Utils module"""
from typing import T, Any, Dict

import torch as tr
import numpy as np
from torch import nn

from .logger import logger


# pylint: disable=too-many-return-statements
def to_tensor(data):
    """Cast data to torch tensor. There is no need for device as `lightning` handles this by itself."""
    if data is None:
        return None
    if isinstance(data, (np.int32, np.int8, np.int16, np.int64, np.float32, np.float64, int, float)):
        return tr.Tensor([data])
    if isinstance(data, list):
        return [to_tensor(x) for x in data]
    if isinstance(data, tuple):
        return tuple(to_tensor(x) for x in data)
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    if isinstance(data, set):
        return {to_tensor(x) for x in data}
    if isinstance(data, tr.Tensor):
        return data
    if isinstance(data, np.ndarray):
        if data.dtype == object:
            return to_tensor(data.tolist())
        return tr.from_numpy(data)
    if callable(data):
        return data
    if isinstance(data, str):
        return data
    raise TypeError(f"Got unknown type: {type(data)}")


def to_device(data, device: tr.device):
    """Moves a generic parameter to the desired torch device."""
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
        if data.dtype == object:
            return to_device(data.tolist(), device)
        return tr.from_numpy(data).to(device)  # pylint: disable=no-member
    if isinstance(data, (int, float, bool, str)):
        return data
    raise TypeError(f"Got unknown type: {type(data)}")


def tr_detach_data(data: T) -> T:
    """Calls detach on compounded torch data"""
    if data is None:
        return None

    if isinstance(data, tr.Tensor):
        return data.detach()

    if isinstance(data, list):
        return [tr_detach_data(x) for x in data]

    if isinstance(data, tuple):
        return tuple(tr_detach_data(x) for x in data)

    if isinstance(data, set):
        return {tr_detach_data(x) for x in data}

    if isinstance(data, dict):
        return {k: tr_detach_data(data[k]) for k in data}

    logger.debug2(f"Got unknown type {type(data)}. Returning as is.")
    return data

def accelerator_params_from_module(module: nn.Module) -> Dict[str, int]:
    """
    Some pytorch lightning madness.
    TODO: multi-device support based on some config params. Could be integrated in TrainSetup using a cfg perhaps.
    """
    # Assume the device based on this.
    device = next(module.parameters()).device
    if device.type == "cuda":
        accelerator = "gpu"
        # cuda:5 => "gpu" and [5]. "cuda" => "gpu" and [0]
        index = [device.index] if isinstance(device.index, int) else [0]
    elif device.type == "cpu":
        # cpu:XX => "cpu" and 1 => this fails otherwise in torch >= 2.0 or lightning >= 2.0
        accelerator = "cpu"
        index = 1
    else:
        assert False, f"Unknown device type: {device}"
    return {"accelerator": accelerator, "devices": index}

def parsed_str_type(item: Any) -> str:
    """Given an object with a type of the format: <class 'A.B.C.D'>, parse it and return 'A.B.C.D'"""
    return str(type(item)).rsplit(".", maxsplit=1)[-1][0:-2]
