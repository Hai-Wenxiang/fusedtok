"""The runnable tour must stay parseable - demo.py is documentation
as well as code, and a broken example shipped in 2.4.1 without any
gate noticing (the CLI checks were only run manually)."""
import ast
import os


def test_demo_py_parses():
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples", "demo.py")
    with open(path, encoding="utf-8") as f:
        ast.parse(f.read())
