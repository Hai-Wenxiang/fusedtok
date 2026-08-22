# Security Policy

## Supported versions

Only the latest release line receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately by opening a GitHub security advisory
(Report a vulnerability button under the Security tab) or by contacting the
maintainer via a private message referencing this repository.

Please do **not** open public issues for suspected vulnerabilities. Include
reproduction details and, where possible, a proof of concept. You will get an
acknowledgment within 7 days.

## Scope notes

fusedtok is a local compute library: it processes tensors you hand it and
performs no network I/O. Security-relevant areas are therefore limited —
primarily memory-safety of the CUDA kernels and the Python/C++ boundary
(shape validation, pointer handling in the zero-copy path).
