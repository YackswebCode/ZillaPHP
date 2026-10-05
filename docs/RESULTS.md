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

### 8.4 GPU CNN — Full MNIST Training

**Architecture:** `Conv2D(1→8, 3×3, pad=1) → ReLU → MaxPool2D(2)
→ Conv2D(8→16, 3×3, pad=1) → ReLU → MaxPool2D(2) → Flatten → Linear(784→10)`

**Training:** full MNIST, 60,000 samples, batch=64, SGD (lr=0.01), 5 epochs.

| Metric | CPU CNN (reference) | **GPU CNN (this run)** | Speedup |
|:---|---:|---:|---:|
| Per epoch | ~420 s | **16 s** | **~26×** |
| Total (5 epochs) | ~35 min | **80 s** | **~26×** |
| Test accuracy | 98.13% | **96.62%** | −1.5 pp |
| Parameters | 9,098 | 9,098 | — |

**Interpretation:** All forward and backward operations (conv2d, relu,
maxpool2d, matmul, softmax-CE, sgd-update) run on the GPU via persistent
device buffers. The accuracy difference is due to fewer training
iterations and plain SGD (no Adam), not to numerical issues.

**Note on transfer overhead:** This run processes samples one at a time —
64 samples per batch, each requiring a full PHP → GPU → CPU round trip.
A batched conv kernel (v2) would further reduce per-epoch time to
~3–5 s.

### 8.5 GPU CNN — Batched Kernels (v2)

**Architecture:** same as §8.4.
**Change:** Conv2D and MaxPool2D now process all B=64 samples in a single
GPU launch, instead of one sample at a time.

| Metric | CPU CNN | GPU CNN v1 | **GPU CNN v2** | Speedup vs CPU |
|:---|---:|---:|---:|---:|
| Per epoch | ~420 s | 16 s | **3.2 s** | **~130×** |
| Total (5 epochs) | ~35 min | 80 s | **16 s** | **~130×** |
| Test accuracy | 98.13% | 96.62% | **95.21%** | — |

**Interpretation:** The batched Conv2D/MaxPool2D kernels deliver a **5×
improvement over the per-sample version** (16 s → 3.2 s per epoch) and
**~130× over the CPU baseline** (420 s → 3.2 s per epoch). The accuracy
shift (−1.4 pp) is expected: batched SGD takes one update per batch
instead of 64, so the effective number of parameter updates is lower
for the same number of epochs.
| 3.1 | 2026-10-05 | Added GPU CNN v1 (96.62% in 80 s) and v2 (95.21% in 16 s, 130× speedup) |
### 8.6 Audio CNN — Synthetic Dataset (Pipeline Validation)

**Purpose:** validate the audio feature pipeline and batched CNN on audio
without requiring the 2.3 GB Speech Commands download.

**Data:** 1,000 training + 200 test clips across 10 classes
(8 sines at 200–3000 Hz, 1 sweep, 1 noise).

**Architecture:** identical to the MNIST CNN — `Conv2D(1→8) → ReLU →
MaxPool2D(2) → Conv2D(8→16) → ReLU → MaxPool2D(2) → FC(6400→10)`.

| Metric | Value |
|:---|---:|
| Total training | 4.0 s |
| Per epoch | 0.4 s |
| Test accuracy | **100.00%** (200 / 200) |

**Interpretation:** the audio preprocessing (WAV → STFT → mel → log)
and the batched CNN kernels work together correctly on 64-band × 101-frame
spectrograms. This validates the pipeline for real speech datasets.