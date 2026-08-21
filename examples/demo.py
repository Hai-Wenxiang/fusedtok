import _fusedtok

x = [float(i) for i in range(8)]

print("input :", x)
print("cpu   :", _fusedtok.axpy(x, 2.0, 1.0))
print("cuda  :", _fusedtok.axpy(x, 2.0, 1.0, cuda=True))
print("gpu   :", _fusedtok.cuda_available())
