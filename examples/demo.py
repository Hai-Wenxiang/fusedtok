"""Minimal demo: run the axpy skeleton on CPU and CUDA and compare.

Run:  py -3.12 examples/demo.py   (with the build dir on PYTHONPATH)
"""

import _fusedtok

x = [float(i) for i in range(8)]

print("input :", x)
print("cpu   :", _fusedtok.axpy(x, 2.0, 1.0))
print("cuda  :", _fusedtok.axpy(x, 2.0, 1.0, cuda=True))
print("gpu   :", _fusedtok.cuda_available())
