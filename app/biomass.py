from __future__ import annotations

import ast
from typing import Iterable

import numpy as np

# Allowed elements when parsing a custom biomass expression.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Attribute,
)
_ALLOWED_SIMPLE_FUNCS = {"abs"}
_ALLOWED_NUMPY_FUNCS = {"log", "log10", "sqrt", "exp", "power", "maximum", "minimum"}


def _validate_formula(expr: str) -> ast.Expression:
    text = (expr or "").strip()
    if not text:
        raise ValueError("Custom biomass equation is empty.")
    tree = ast.parse(text, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported element '{type(node).__name__}' in biomass equation.")
        if isinstance(node, ast.Name):
            if node.id not in {"x", "np"} | _ALLOWED_SIMPLE_FUNCS | _ALLOWED_NUMPY_FUNCS:
                raise ValueError(f"Name '{node.id}' is not allowed in biomass equation.")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _ALLOWED_SIMPLE_FUNCS and func.id not in _ALLOWED_NUMPY_FUNCS:
                    raise ValueError(f"Function '{func.id}' is not allowed in biomass equation.")
            elif isinstance(func, ast.Attribute):
                if not (isinstance(func.value, ast.Name) and func.value.id == "np" and func.attr in _ALLOWED_NUMPY_FUNCS):
                    raise ValueError("Only numpy helpers like np.log or np.sqrt are allowed.")
            else:
                raise ValueError("Unsupported function call in biomass equation.")
    return tree

def fw_biocapped(arr_cm2):
    A_cal_max = 500  # cm² (max area used in calibration)

    # Original model
    fw_pred = 0.014 * np.power(arr_cm2, 1.65)

    # Biological density cap from calibration limit
    fw_cal_max = 0.014 * (A_cal_max ** 1.65)
    rho_max = fw_cal_max / A_cal_max  # g/cm²

    # Maximum allowed FW based on capped density
    fw_cap = rho_max * arr_cm2

    # Apply cap
    fw_corrected = np.minimum(fw_pred, fw_cap)

    return fw_corrected

def _eval_custom_formula(expr: str, area_cm2: np.ndarray) -> np.ndarray:
    x = np.asarray(area_cm2, dtype=np.float64)
    tree = _validate_formula(expr)
    env = {"x": x, "np": np, "abs": np.abs, "power": np.power}
    env.update({name: getattr(np, name) for name in _ALLOWED_NUMPY_FUNCS if hasattr(np, name)})
    code = compile(tree, "<biomass_formula>", "eval")
    try:
        res = eval(code, {"__builtins__": {}}, env)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to evaluate biomass equation: {exc}") from exc
    arr = np.asarray(res, dtype=np.float64)
    target_shape = x.shape
    try:
        arr = np.broadcast_to(arr, target_shape if target_shape else tuple())
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Biomass equation result cannot be broadcast to match the input area.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("Biomass equation returned NaN or infinite values.")
    return arr


def biomass_from_area_cm2(
    area_cm2: Iterable[float] | np.ndarray | float,
    model: str = "madagascar",
    custom_formula: str | None = None,
) -> np.ndarray:
    """
    Compute biomass in grams from an area expressed in cm^2.

    - Presets:
      - madagascar: quadratic relationship.
      - indonesia: Nurdin et al. 2023.
    - Custom: supply `model="custom"` and an equation using `x` (plot area in cm^2).
    """
    arr = np.asarray(area_cm2, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    key = (model or "madagascar").strip().lower()
    if key == "custom":
        if not custom_formula or not str(custom_formula).strip():
            raise ValueError("Provide a biomass equation when using the custom model.")
        return _eval_custom_formula(str(custom_formula), arr)
    if key == "indonesia":
        # Nurdin et al. 2023: biomass(g) = 0.014 * area(cm^2) ^ 1.65
        with np.errstate(invalid="ignore"):
            return fw_biocapped(arr)
            # return 0.014 * 1.65 * np.power(arr, 0.65)
    if key in {"madagascar_poly", "madagascar_quadratic"}:
        # return (-0.00057 * arr * arr) + (0.23244 * arr) + 13.2624
        return 0.19982 * arr ** 1.15378   # Default: Madagascar linear relationship
    return 0.5621 * arr


def biomass_from_area_m2(
    area_m2: Iterable[float] | np.ndarray | float,
    model: str = "madagascar",
    custom_formula: str | None = None,
) -> np.ndarray:
    """Helper variant that accepts area in square meters."""
    arr = np.asarray(area_m2, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    return biomass_from_area_cm2(arr * 10000.0, model=model, custom_formula=custom_formula)
