"""Dataclasses -> JSON-safe dicts. Image arrays are dropped (served by their own endpoints)."""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
               if not isinstance(getattr(obj, f.name), np.ndarray)}
        for name in ("needs_review", "percentage", "total", "max_total"):  # useful computed properties
            if hasattr(type(obj), name) and isinstance(getattr(type(obj), name), property):
                out[name] = to_jsonable(getattr(obj, name))
        return out
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return None
    return obj
