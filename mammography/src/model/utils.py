from inspect import signature
from typing import Any, Dict

import torch


def replace_layer(layer_to_replace: torch.nn.Module, **new_layer_kwargs) -> torch.nn.Module:
    class_signature = signature(layer_to_replace.__class__).parameters
    layer_params = {k: getattr(layer_to_replace, k) for k in class_signature.keys() if hasattr(layer_to_replace, k)}
    layer_params.update(new_layer_kwargs)
    if "bias" in layer_params.keys() and isinstance(layer_params["bias"], torch.Tensor):
        layer_params["bias"] = True
    return type(layer_to_replace)(**layer_params)


def replace_submodules(module: torch.nn.Module, **replacement_kwargs: Dict[str, Any]) -> torch.nn.Module:
    for target, replacement in replacement_kwargs.items():
        parent_name, _, child_name = target.rpartition(".")
        parent = module.get_submodule(parent_name)
        child = parent.get_submodule(child_name)
        new_child = replace_layer(child, **replacement)
        setattr(parent, child_name, new_child)
    return module
