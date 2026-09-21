# ZillaPHP — Experimental Results

**Document version:** 1.1
**Date:** 2026-09-21
**Framework version:** ZillaPHP 0.1.0-dev
**Reference architecture:** *ZillaPHP: A Comprehensive PHP-Native Framework for Machine Learning and Artificial Intelligence* (Yahaya Ibrahim, 2025)

---

## Abstract

This document reports the first complete set of empirical results for ZillaPHP.
It covers (i) numerical correctness of reverse-mode automatic differentiation,
(ii) CPU performance of native C kernels versus pure PHP,
(iii) GPU performance on a virtualized NVIDIA T4 via PHP FFI + CUDA,
and (iv) end-to-end training results on MNIST and a character-level
Shakespeare language model.

All measurements are reproducible from the source repository using the
commands cited in each section. This document does not claim superiority
over established frameworks — it reports what a PHP-first framework
delegating to native backends can achieve in practice.

---

## 1. Hardware and Software Environment

### Primary development machine

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core (4 cores, AVX2) |
| RAM | 8 GB |
| GPU | Intel HD Graphics 620 (not used) |
| OS | Linux Mint |
| PHP | 8.4.25 (NTS) |
| Compiler | GCC (with `-O3 -march=native -ffast-math`) |
| BLAS | OpenBLAS 0.3.26 (`OPENBLAS_NUM_THREADS=1`) |

### GPU environment (Colab)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla T4 (15 GB) |
| Driver | 580.82.07 |
| CUDA | 13.0 (runtime 12.8) |
| Virtualization | GRID (Google Colab) |

---

## 2. Numerical Correctness

Reverse-mode autograd is verified against central finite differences
for every differentiable operation. Tolerances are matched to float32
precision (`eps = 1e-3`, tolerance `1e-2`, per PyTorch's float32 gradcheck).

### 2.1 Gradient verification results

| Operation | Verdict | Notes |
|-----------|---------|-------|
| `add`, `sub`, `mul`, `div` | ✅ pass | with broadcasting |
| `matmul` | ✅ pass | verified on 2×2 case |
| `transpose` | ✅ pass | |
| `sum`, `mean` | ✅ pass | |
| `relu` | ✅ pass | subgradient at 0 chosen as 0 |
| `exp`, `log` | ✅ pass | |
| `softmax` | ✅ pass | |
| `layerNorm` | ✅ pass | hand-written backward |
| `maskedFill` | ✅ pass | |
| `chunkColumns` / `concatColumns` | ✅ pass | scatter/gather gradients |
| `conv2d` (input, weight, bias) | ✅ pass | verified on 4×4 with padding |
| `maxPool2d` | ✅ pass | argmax-based scatter |

**Test suite status:** `OK (11 tests, 13 assertions)` — see `tests/Numerical/`.

### 2.2 Representative gradient check

For `f(x) = x^2` at `x = -3`:
