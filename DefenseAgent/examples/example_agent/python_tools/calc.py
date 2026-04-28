"""A safe arithmetic calculator for the example agent.

Wired into the agent via `DefenseAgent/examples/example_agent/profile.yaml`:

    tools:
      python:
        - python_tools/calc.py:calculator

The function's type hints + docstring become the tool's input schema +
description automatically — the LLM sees this exactly.
"""
import ast
import math
import operator


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "min": min, "max": max,
}


def calculator(expression: str) -> str:
    """Evaluate a Python-style arithmetic expression. Supports + - * / // % **, unary +/-, and the functions sqrt, log, log10, exp, sin, cos, tan, abs, round, min, max. Returns the numeric result as a string, or an error message."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"calculator error: {type(e).__name__}: {e}"


def _eval_node(node: ast.AST) -> float:
    """Walk the parsed AST, allowing only the whitelisted ops and functions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"unsupported expression node: {ast.dump(node)}")
