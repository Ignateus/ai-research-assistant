"""Safe math expression evaluator — no eval(), uses Python's AST."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

# Allowed operators
_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Allowed math constants and functions
_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _eval_node(node: ast.AST) -> float:
    match node:
        case ast.Constant(value=v) if isinstance(v, (int, float)):
            return v
        case ast.Name(id=name) if name in _SAFE_NAMES:
            return _SAFE_NAMES[name]
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _OPERATORS:
            return _OPERATORS[type(op)](_eval_node(left), _eval_node(right))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _OPERATORS:
            return _OPERATORS[type(op)](_eval_node(operand))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _SAFE_NAMES:
            fn = _SAFE_NAMES[name]
            return fn(*[_eval_node(a) for a in args])
        case _:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression and return the result as a string.

    Args:
        expression: A math expression, e.g. "2 ** 10", "sqrt(144)", "log(e)"

    Returns:
        The numeric result as a string, or an error message.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        # Return clean int string if result is a whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 10))
    except ZeroDivisionError:
        return "Error: division by zero"
    except ValueError as exc:
        return f"Error: {exc}"
    except SyntaxError:
        return f"Error: invalid expression '{expression}'"


# --- Tool definition (Anthropic schema) ---

TOOL_DEFINITION = {
    "name": "calculator",
    "description": (
        "Evaluate a mathematical expression. Supports +, -, *, /, **, %, //, "
        "and functions: sqrt, log, log10, sin, cos, tan, abs, round, ceil, floor. "
        "Constants: pi, e."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'.",
            }
        },
        "required": ["expression"],
    },
}
