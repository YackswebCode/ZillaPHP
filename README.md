# ZillaPHP

**A PHP-native framework for machine learning and artificial intelligence.**

Build, train, and run neural networks directly from PHP — with a pure-PHP API, an extensible backend system, and native C acceleration through FFI.

Text · Audio · Image · Video

---

## Status

> **⚠️ Experimental — under active development.**
>
> ZillaPHP is a research and engineering project, not a production-ready
> replacement for PyTorch or TensorFlow. APIs may change before 1.0.
> Performance claims come from reproducible benchmarks (see [`benchmarks/`](benchmarks/)).

| Milestone | Status |
|-----------|--------|
| Tensor engine (shape, stride, dtype, device) | ✅ Working |
| Reverse-mode autograd with gradient checking | ✅ Working |
| Neural network modules (`Linear`, `ReLU`, `Sequential`) | ✅ Working |
| Losses (`MSE`, `CrossEntropy`) | ✅ Working |
| Optimizers (`SGD`, `Adam`, `AdamW`) | ✅ Working |
| Batched training (`DataLoader`, `Trainer::fitLoader`) | ✅ Working |
| Model checkpointing (JSON, SafeTensors planned) | ✅ Working |
| Native C++ CPU backend via PHP FFI | ✅ Working |
| Real MNIST training | ✅ 95.6% (10k subset) |
| CUDA backend | 🔜 Planned |
| ROCm backend | 🔜 Planned |
| Transformer / attention / tokenizer | 🔜 In progress |
| Multimodal (audio, image, video) | 🔜 Planned |

---

## Why ZillaPHP Exists

Modern machine learning frameworks are concentrated around Python and C++.
Meanwhile, PHP powers a huge portion of the web — CMSes, APIs, financial
systems, enterprise applications. When those applications need AI, developers
are forced to introduce another ecosystem: a Python microservice, a REST API,
or a binding layer.

ZillaPHP proposes a different approach: **PHP as the primary framework
language**, with performance-critical computation delegated to optimized native
backends (C, and later CUDA / ROCm).

The architectural principle:

    PHP Model  →  Tensor API  →  Runtime  →  Operator Dispatcher  →  Backend
                                                                       ↓
                                                        CPU (C) / CUDA / ROCm

The developer writes PHP. The backend handles numerical execution.

---

## Features

- **PHP-first API** — `$model->forward($x)`, `$loss->backward()`, `$optimizer->step()`
- **Reverse-mode autograd** — dynamic computation graph, verified against finite differences
- **Broadcasting** — `[N, D] + [D]` works the way it should
- **Batched training** — 10–30× faster than per-sample loops
- **Native C kernels via FFI** — up to **122× faster matmul** than pure PHP
- **Extensible backend contract** — implement one interface to add a new device
- **CLI** — `zilla doctor`, `zilla train`, `zilla device:list`
- **Honest maturity** — experimental features are clearly marked

### Current native kernel speedups (Acer TravelMate, 4 cores)

| Operation | Pure PHP | Native C | Speedup |
|-----------|----------|----------|---------|
| matmul 64×64 | 44 ms | 2.0 ms | **22×** |
| matmul 128×128 | 250 ms | 8.3 ms | **30×** |
| matmul 256×256 | 1622 ms | 29 ms | **56×** |
| matmul 512×512 | 16077 ms | 132 ms | **122×** |

Full MNIST (60,000 samples, MLP 784→256→128→10): **~5 min / epoch** on CPU.

---

## Requirements

- **PHP 8.4+** with `FFI` enabled
- **C compiler** (GCC or Clang)
- **Composer**
- Linux / macOS (Windows works, not yet tested)

---

## Installation

### 1. Clone

```bash
git clone https://github.com/YackswebCode/ZillaPHP.git
cd ZillaPHP# ZillaPHP
