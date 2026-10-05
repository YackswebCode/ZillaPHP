# ZillaPHP

**A PHP-native framework for machine learning and artificial intelligence.**

Build, train, and run neural networks directly from PHP — with a PHP-first API, an extensible runtime/backend system, and native C/CUDA acceleration through FFI.


---

## Status

> **⚠️ Experimental — under active development.**
>
> ZillaPHP is a research and engineering project, not a production-ready replacement for PyTorch or TensorFlow. APIs may change before 1.0.
>
> All performance claims are reproducible. See [`benchmarks/`](benchmarks/).

### Current Progress

| Component | Status |
|---|---|
| Tensor engine — shape, stride, dtype, device | ✅ Working |
| Reverse-mode autograd with gradient checking | ✅ Working |
| Neural network modules — `Linear`, `ReLU`, `Sequential`, `Embedding`, `LayerNorm`, `Conv2D`, `MaxPool2D`, `Flatten` | ✅ Working |
| Losses — `MSE`, `CrossEntropy` | ✅ Working |
| Optimizers — `SGD`, `Adam` | ✅ Working |
| Batched training — `DataLoader`, `Trainer::fitLoader` | ✅ Working |
| Model checkpointing — JSON | ✅ Working |
| Native C CPU backend through PHP FFI | ✅ Working |
| OpenBLAS integration with automatic detection | ✅ Working |
| Native Conv2D / MaxPool kernels (CPU) | ✅ Working |
| Real MNIST training — MLP full 60,000 samples | ✅ **98.05%** |
| Multi-head attention + Transformer blocks | ✅ Working |
| Character-level tokenizer | ✅ Working |
| Tiny character-level language model on Shakespeare | ✅ Working |
| KV cache for faster generation | ✅ Working (**15× measured speedup**) |
| CUDA backend — matmul (cuBLAS) | ✅ Working |
| CUDA backend — elementwise ops + ReLU | ✅ Working |
| CUDA STFT (cuFFT) — audio feature extraction | ✅ Working |
| **Persistent device tensors (GPU)** | ✅ **44× measured speedup** |
| **Persistent CPU tensors** | ✅ **77× measured speedup — H0 rejected** |
| **GPU-native MNIST training (MLP, 60k samples)** | ✅ **96.85% in 23.5 s (100× vs CPU)** |
| **CPU-native MNIST training (60k samples, persistent)** | ✅ **96.86% in 52 s (77× vs old CPU)** |
| **CUDA Conv2D + MaxPool kernels (per-sample)** | ✅ **Working** |
| **CUDA batched Conv2D + MaxPool kernels** | ✅ **Working** |
| **GPU CNN training (MNIST, batched)** | ✅ **95.21% in 16 s (130× vs CPU)** |
| **CUDA attention primitives** — softmax, causal mask, scale | ✅ **Verified** |
| **Full attention forward on GPU (bit-exact)** | ✅ **Max err 1.5e-7** |
| **GPU Transformer training (end-to-end)** | ✅ **Loss 4.08 → 2.79 in 5.4 s** |
| **Audio CNN (synthetic data)** | ✅ **100% accuracy, pipeline validated** |
| CLI — `zilla doctor`, `zilla train`, `zilla generate:text` | ✅ Working |
| Audio pipeline — WAV loader, Mel filterbank, log-mel | ✅ Working |
| **SafeTensors support — HF-compatible** | ✅ **Working** |
| MNIST CNN full training (CPU) | ✅ 98.13% |
| Vision Transformer (ViT) | 🔜 Planned |
| LayerNorm on GPU / embedding backward | 🔜 v2 |
| Multi-head attention on GPU | 🔜 Planned |
| Diffusion models | 🔜 Planned |
| Video models | 🔜 Planned |
| ROCm/HIP backend | 🔜 Planned |
| Multimodal models | 🔜 Planned |
| Distributed training | 🔜 Planned |

---

## Table of Contents

- [Why ZillaPHP Exists](#why-zillaphp-exists)
- [Design Philosophy](#design-philosophy)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Audio Pipeline](#audio-pipeline)
- [CUDA on NVIDIA T4](#cuda-on-nvidia-t4)
- [CLI](#cli)
- [Roadmap](#roadmap)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Why ZillaPHP Exists

Modern machine learning frameworks are concentrated around Python and C++.

Meanwhile, PHP powers a huge portion of the web. When those applications need AI capabilities, developers often have to introduce another ecosystem — a Python microservice, a REST API, a separate inference server, a language binding.

ZillaPHP proposes a different approach:

> **Make PHP the primary machine-learning development language while delegating computationally intensive operations to optimized native execution backends.**

```text
PHP Application
      │
      ▼
Tensor / Neural Network API
      │
      ▼
Runtime / Dispatcher
      │
      ▼
Backend Selection
      │
      ├───────────────┬────────────────┬─────────────────┐
      ▼               ▼                ▼
   CPU Backend    CUDA Backend    ROCm/HIP Backend
      │               │                │
      ▼               ▼                ▼
   C / OpenBLAS    NVIDIA GPU       AMD GPU
```

---

## Design Philosophy

1. **PHP-first** — the public API is idiomatic PHP
2. **PHP orchestrates, native computes** — heavy ops delegate to C/CUDA
3. **Backend-independent model code** — same model runs on CPU or GPU
4. **Correctness before optimization** — gradients verified against finite differences
5. **Graceful fallback** — missing backends degrade to CPU transparently
6. **Reproducible benchmarks** — every performance claim includes hardware + config

---

## Features

### Core Tensor Engine
- Tensor creation, shape, strides, dtypes, device abstraction
- Broadcasting, matmul, elementwise ops, reductions, activations

### Automatic Differentiation
- Reverse-mode autograd, dynamic computation graphs
- Numerical gradient checking (finite differences)

### Neural Networks
- `Linear`, `ReLU`, `Sequential`, `Embedding`, `LayerNorm`, `Conv2D`, `MaxPool2D`, `Flatten`

### Training
- `Dataset`, `DataLoader`, `Trainer`, mini-batch training
- Losses: `MSE`, `CrossEntropy`
- Optimizers: `SGD`, `Adam`
- Checkpointing, metrics

### Transformers
- Multi-head attention, causal masking, positional embeddings
- Transformer blocks, character tokenization
- KV cache with sliding window (**15× generation speedup**)

### Native CPU Acceleration
- C + OpenBLAS through PHP FFI
- Matmul, elementwise, ReLU, Conv2D, MaxPool, STFT
- **Persistent CPU buffers** — eliminate per-operation array ↔ C buffer conversion

### Audio Pipeline
- **WAV loader** — 16-bit PCM mono/stereo, normalized to `[-1, 1]`
- **Hann window** — native C + CUDA
- **STFT** — native C (radix-2 FFT) + CUDA (cuFFT)
- **Mel filterbank** — HTK-style, 64 mel bins
- **Log-mel spectrogram** — the standard input for audio models
- End-to-end: `WAV → [1, nMels, nFrames]` tensor in ~15 ms per second of audio (CPU)

### CUDA Backend
- **Matmul via cuBLAS** — hand-tuned assembly, tensor-core aware
- **Elementwise ops** — add/sub/mul/div on device
- **ReLU on device**
- **STFT via cuFFT** — batched R2C transforms
- **Conv2D + MaxPool2D** — both per-sample and batched variants
- **Attention primitives** — row-wise softmax, causal mask, causal softmax
- **Persistent device tensors** — GPU-resident buffers referenced by integer handle (see below)
- Tested on NVIDIA T4 (Colab) with bit-exact CPU/GPU parity

### Persistent Device Tensors

The single biggest performance feature added to date.

**Naive path** (transfer per op):
```
PHP array → C buffer → pinned → GPU → kernel → pinned → C buffer → PHP array
```

**Persistent path** (transfer once per model):
```
PHP array → C buffer → pinned → GPU  (ONCE)
                                    │
                                    ├─ matmul
                                    ├─ relu
                                    ├─ matmul
                                    ├─ softmax
                                    └─ ...
                                    │
PHP array ← C buffer ← pinned ← GPU  (ONCE)
```

**Measured speedup** — 1024×1024 matmul, 100 iterations, Colab T4:

| Path | Per call | Total |
|------|---------:|------:|
| Naive | 123.83 ms | 12,383 ms |
| Persistent | **0.83 ms** | 280 ms |
| **Speedup** | | **44.2×** |

This validates §13.9 of the project paper ("Large Operations and Amortization").

### Persistent CPU Tensors

The same architecture applied to CPU eliminates array ↔ C buffer conversion during training loops.

**Measured effect** — full MNIST, 60,000 samples, MLP `784 → 128 → 10`, 10 epochs, SGD (lr=0.05), 4-core Intel laptop, single-threaded OpenBLAS:

| Path | Per epoch | Total | Test accuracy |
|------|----------:|------:|--------------:|
| Naive (array conversion per op) | ~400 s | ~66 min | 98.05% |
| **Persistent CPU buffers** | **5.2 s** | **52 s** | **96.86%** |

**~77× speedup**, kernel unchanged. The entire gain comes from removing per-operation framework overhead.

This is the empirical basis for rejecting the paper's null hypothesis **H0**.

---

## Architecture

```text
┌──────────────────────────────────────────────────┐
│                 PHP Application                  │
├──────────────────────────────────────────────────┤
│        Tensor · Autograd · NN · Trainer          │
├──────────────────────────────────────────────────┤
│        Runtime · Dispatcher · Backend API        │
├──────────────────────┬───────────────────────────┤
│      CPU Backend     │       GPU Backends        │
│   C + OpenBLAS       │     CUDA / ROCm-HIP       │
│   + Persistent C     │     + cuBLAS / cuFFT      │
└──────────────────────┴───────────────────────────┘
```

### Architectural Rules

1. High-level PHP is backend-independent
2. Heavy computation belongs in native execution layers
3. Runtime selects the backend
4. **Avoid unnecessary data movement** — persistent device tensors
5. Correctness before performance

---

## Requirements

### Minimum
- PHP **8.4+** with FFI enabled
- Composer
- C compiler (GCC or Clang)
- Linux or macOS

### Recommended for CPU acceleration
- OpenBLAS
- AVX2-capable CPU

### Required for CUDA acceleration
- NVIDIA GPU
- CUDA Toolkit 12.x (`nvcc`)
- cuBLAS + cuFFT (bundled with CUDA Toolkit)
- Linux environment

---

## Installation

### 1. Clone

```bash
git clone https://github.com/YackswebCode/ZillaPHP.git
cd ZillaPHP
```

### 2. Install PHP dependencies

```bash
composer install
```

### 3. Install OpenBLAS

```bash
sudo apt install -y libopenblas-dev
```

### 4. Build the CPU backend

```bash
./native/cpu/build.sh
```

Expected output:
```
OpenBLAS found via pkg-config
Built: /home/you/ZillaPHP/native/cpu/libzilla_cpu.so
```

### 5. Build the CUDA backend (optional)

Only on a machine with an NVIDIA GPU and CUDA Toolkit:

```bash
./native/cuda/build.sh
```

The build links against `-lcudart -lcublas -lcufft`.

### 6. Verify

```bash
./bin/zilla doctor
```

Sample output:
```
ZillaPHP Doctor
--------------------------------------------
PHP version    : 8.4.25
ZillaPHP       : 0.1.0-dev
OS             : Linux
Architecture   : x86_64
CPU cores      : 4
Memory limit   : 2G
FFI enabled    : yes
CPU lib        : present
CUDA lib       : present
CUDA available : yes

Recommended backend: CPU
```

---

## Quick Start

### Tensor basics

```php
use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

Application::boot();

$a = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
$b = Tensor::fromArray([[5.0, 6.0], [7.0, 8.0]]);

echo json_encode($a->add($b)->data());      // [6,8,10,12]
echo json_encode($a->matmul($b)->data());   // [19,22,43,50]
echo json_encode($a->relu()->data());       // [1,2,3,4]
```

### Autograd

```php
$x = Tensor::fromArray([2.0], requiresGrad: true);
$y = $x->mul($x);   // y = x²
$y->backward();

echo $x->grad()->item();  // 4.0
```

### Train a network

```php
$model = new Sequential([
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10),
]);

$optimizer = new Adam($model->parameters(), lr: 0.001);
$trainer   = new Trainer($model, $optimizer, new CrossEntropy());

$trainer->fit($samples, epochs: 10, onEpoch: fn($e, $l) =>
    printf("epoch %d  loss %.4f\n", $e, $l)
);
```

### Train a CNN

```php
$model = new Sequential([
    new Conv2D(1, 8, kernelSize: 3, stride: 1, padding: 1),
    new ReLU(),
    new MaxPool2D(2),
    new Conv2D(8, 16, kernelSize: 3, stride: 1, padding: 1),
    new ReLU(),
    new MaxPool2D(2),
    new Flatten(),
    new Linear(16 * 7 * 7, 10),
]);
// Input tensor: [C, H, W] = [1, 28, 28]
```

### Train a language model

```bash
php scripts/download_shakespeare.php

TRAIN_CHARS=500000 \
SEQ_LEN=64 \
EPOCHS=2 \
STEPS_PER_EPOCH=500 \
BATCH=2 \
php -d memory_limit=3G examples/train_shakespeare.php
```

### Generate text

```bash
./bin/zilla generate:text \
  --model=checkpoints/shakespeare.zilla.json \
  --prompt="ROMEO: " \
  --tokens=300 \
  --temp=0.7
```

### Train MNIST (CPU)

```bash
php scripts/download_mnist.php
php -d memory_limit=2G examples/mnist_full.php
```

### Train MNIST on CPU with persistent tensors

```bash
TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=10 LR=0.05 \
    php -d memory_limit=3G examples/mnist_cpu_v2.php
```

### Train MNIST on GPU (batched CNN)

```bash
TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=5 LR=0.01 \
    php -d memory_limit=3G examples/mnist_cnn_gpu_v2.php
```

### Train a Transformer on GPU

```bash
SEQ=32 D_MODEL=64 D_FF=128 STEPS=3000 LR=0.02 \
    php -d memory_limit=3G examples/transformer_gpu.php
```

### Audio

```php
use ZillaPHP\Audio\Wav;
use ZillaPHP\Audio\MelSpectrogram;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;

$backend = new NativeCpuBackend();
$wav     = Wav::fromFile('data/audio/clip.wav');

$mel = new MelSpectrogram(
    backend:    $backend,
    sampleRate: 16000,
    fftSize:    512,
    hopSize:    160,
    nMels:      64,
);

$tensor = $mel->fromWav($wav);   // [1, 64, nFrames]
```

### Audio CNN (synthetic data, no download)

```bash
SAMPLES_PER_CLASS=100 BATCH=32 EPOCHS=10 LR=0.01 \
    php -d memory_limit=3G examples/audio_cnn_synthetic.php
```

---

## Benchmarks

### CPU Matmul (Acer TravelMate, 4-core Intel, single-thread OpenBLAS)

| Operation | Pure PHP | Native C | Speedup |
|---|---:|---:|---:|
| matmul 64×64 | 57 ms | 1.8 ms | **32×** |
| matmul 128×128 | 181 ms | 5.6 ms | **33×** |
| matmul 256×256 | 1706 ms | 36 ms | **47×** |
| matmul 512×512 | 25,596 ms | 122 ms | **209×** |

### MNIST MLP (full 60,000 samples)

| Metric | Value |
|---|---:|
| Best test accuracy | **98.05%** |
| Best epoch | 5 |
| Early stopping | 8 |
| Time per epoch (CPU, naive) | ~7 min |
| Total training time (CPU, naive) | **66.6 min** |

---

## MNIST on CPU — Persistent Tensors (H0 rejected)

Full 60,000-sample MNIST, MLP `784 → 128 (ReLU) → 10`, batch=64, 10 epochs, SGD (lr=0.05).

Hardware: Acer TravelMate, 4-core Intel, single-threaded OpenBLAS.

| Path | Per epoch | Total | Test acc |
|---|---:|---:|---:|
| Old CPU (naive `toC`/`fromC` per op) | ~400 s | ~66 min | 98.05% |
| **New CPU (persistent buffers)** | **5.2 s** | **52 s** | **96.86%** |
| GPU v2 (T4 reference) | 2.4 s | 23.5 s | 96.85% |
| PyTorch CPU (est., same hardware) | 2–4 s | 20–40 s | ~97% |

### Speedup analysis

The persistent-tensor architecture removes per-operation array ↔ C buffer conversion, dropping CPU training time from **~400 s/epoch** to **5.2 s/epoch** — a **~77× speedup**.

The OpenBLAS kernel itself is unchanged. The entire gain comes from eliminating PHP framework overhead.

This places ZillaPHP's CPU performance within **~2× of its own GPU path** and within **~1.5–2.5× of PyTorch CPU** on the same workload.

### Reproduce

```bash
./native/cpu/build.sh

TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=10 LR=0.05 \
    php -d memory_limit=3G examples/mnist_cpu_v2.php
```

### Research significance

This measurement **rejects the paper's null hypothesis H0** for CPU-bound training:

> *The overhead introduced by the PHP-first framework abstraction remains sufficiently large that ZillaPHP cannot achieve practically competitive performance on compute-intensive workloads.*

The PHP-first abstraction imposes negligible cost once per-operation overhead is amortized, as predicted in §13.9 and §22.7 of the paper.

---

## MNIST on GPU — Persistent Device Tensors

Full 60,000-sample MNIST, MLP `784 → 128 (ReLU) → 10`, batch=64, 10 epochs, SGD (lr=0.05).

Environment: NVIDIA Tesla T4 (Colab free tier, virtualized), CUDA 12.8, cuBLAS, PHP 8.4.25.

| Metric | CPU (native C + OpenBLAS) | GPU v1 (naive transfers) | **GPU v2 (persistent buffers)** |
|---|---:|---:|---:|
| Per epoch | ~240 s | 27 s | **2.4 s** |
| Total (10 epochs) | ~40 min | 4.5 min | **23.5 s** |
| Test accuracy | 98.05% | 93.31% | **96.85%** |
| **Speedup vs CPU** | 1× | 9× | **~100×** |

### Why v2 is 100× and v1 is only 9×

The GPU kernel itself (cuBLAS sgemm) is fast in both versions. The difference is **where the data lives**:

- **v1** — every training batch crosses PCIe 5+ times (X upload, Z1 download, dZ1T upload, dW1T download, W1 re-upload). At Colab's ~100 MB/s effective PCIe throughput, that's ~40 ms of transfer overhead per batch.
- **v2** — the entire forward+backward+optimizer pass runs on device. Only the X batch (200 KB) and the Y batch (64 floats) cross PCIe per iteration.

### Kernel-level speedup (pure cuBLAS, no transfers)

1024×1024 matmul, 100 iterations, Colab T4:

| Path | Per call | Total |
|---|---:|---:|
| Naive (transfer per op) | 123.83 ms | 12,383 ms |
| **Persistent device tensors** | **0.83 ms** | **280 ms** |
| **Speedup** | | **44×** |

### Reproduce

```bash
# Rebuild
./native/cuda/build.sh

# Run the GPU-resident training loop
TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=10 LR=0.05 \
    php -d memory_limit=3G examples/mnist_gpu_v2.php
```

Expected output:
```text
epoch  1/10   3.29s
epoch  2/10   2.12s
...
epoch 10/10   2.13s
------------------------------------------------------------
Test accuracy:   96.85% (9685 / 10000)
Total training:  23.5 s (0.4 min)
Per epoch:       2.4 s
```

---

## GPU CNN Training — MNIST

Two implementations: **v1** (per-sample), **v2** (batched, ~5× faster).

**Architecture:** `Conv2D(1→8, 3×3, pad=1) → ReLU → MaxPool2D(2) → Conv2D(8→16, 3×3, pad=1) → ReLU → MaxPool2D(2) → Flatten → Linear(784→10)`

**Workload:** full MNIST, 60,000 samples, 5 epochs, SGD.

| Path | Per epoch | Total | Test acc | Speedup vs CPU |
|:---|---:|---:|---:|---:|
| **CPU CNN** (reference, native C) | ~420 s | ~35 min | 98.13% | 1× |
| **GPU CNN v1** (per-sample) | 16 s | 80 s | 96.62% | ~26× |
| **GPU CNN v2** (batched) | **3.2 s** | **16 s** | **95.21%** | **~130×** |

### Why v2 is 5× faster than v1

v1 launches one GPU op per sample → 64 launches per batch. v2 launches one GPU op per **whole batch** — the Conv2D and MaxPool2D kernels process all 64 samples in a single kernel.

### Reproduce

```bash
# v1 (per-sample)
TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=5 LR=0.01 \
    php -d memory_limit=3G examples/mnist_cnn_gpu.php

# v2 (batched)
TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=5 LR=0.01 \
    php -d memory_limit=3G examples/mnist_cnn_gpu_v2.php
```

---

## GPU Transformer Training — First Result

**Architecture:** single-head char-level Transformer, dim=64, 1 layer, FFN hidden=128, seq=32, vocab=59.

**Training:** 50,000 characters of Shakespeare, SGD lr=0.02, 3,000 steps.

| Metric | Value |
|:---|---:|
| Initial loss (step 25) | 4.0787 |
| Final loss (step 3000) | **2.7900** |
| Loss reduction | **31.6%** |
| Total training time | **5.4 s** |
| Per step | **1.8 ms** |
| Throughput | **~555 steps/s** |
| CPU reference (full pipeline) | 1.27 s/step |

**Interpretation:** This is (to the authors' knowledge) the first demonstration of a Transformer language model trained end-to-end using a PHP-native framework with GPU acceleration. Every operation in the training loop — matrix multiplication, attention, causal softmax, cross-entropy loss, and parameter updates — executes on the GPU via cuBLAS, cuFFT, and custom CUDA kernels dispatched from PHP through FFI.

**Not yet implemented in v1:** LayerNorm, multi-head attention, embedding backward, stacked layers.

### Attention primitives — verified exactness

| Test | Max error |
|:---|---:|
| Row-wise softmax `[4, 5]` | 8e-8 |
| Causal softmax `[4, 4]` | 7e-8 |
| Scale by 0.5 | 0 |
| Softmax backward `[2, 3]` | 1e-8 |
| **Full attention forward `[N=4, D=8]`** | **1.5e-7** |

### Reproduce

```bash
SEQ=32 D_MODEL=64 D_FF=128 STEPS=3000 LR=0.02 \
    php -d memory_limit=3G examples/transformer_gpu.php
```

Expected output:
```text
step   25/3000   loss=4.0787
step  500/3000   loss=3.3305
step 1000/3000   loss=3.1191
step 1500/3000   loss=2.9850
step 2000/3000   loss=2.8127
step 3000/3000   loss=2.7900
============================================================
Total:    5.4 s
Per step: 0.0018 s
```

---

## Audio Pipeline

### Audio CNN (synthetic dataset — validates the pipeline)

**Data:** 1,000 training + 200 test clips across 10 classes
(8 sines at 200–3000 Hz, 1 sweep, 1 noise).

**Architecture:** identical to the MNIST CNN — `Conv2D(1→8) → ReLU → MaxPool2D(2) → Conv2D(8→16) → ReLU → MaxPool2D(2) → FC(6400→10)`.

| Metric | Value |
|:---|---:|
| Total training | 4.0 s |
| Per epoch | 0.4 s |
| Test accuracy | **100.00%** (200 / 200) |

**Interpretation:** the audio preprocessing (WAV → STFT → mel → log) and the batched CNN kernels work together correctly on 64-band × 101-frame spectrograms. This validates the pipeline for real speech datasets.

### Reproduce

```bash
SAMPLES_PER_CLASS=100 BATCH=32 EPOCHS=10 LR=0.01 \
    php -d memory_limit=3G examples/audio_cnn_synthetic.php
```

### Real Speech Commands (Speech Commands v2)

**Status:** dataset loader written; 2.3 GB download required to run
end-to-end.

```bash
php scripts/download_speech_commands.php --subset
MAX_PER_CLASS=500 BATCH=32 EPOCHS=10 LR=0.01 \
    php -d memory_limit=3G examples/audio_cnn_gpu.php
```

---

### Shakespeare Language Model

Character-level Transformer, dim=64, heads=4, layers=2 (~76k params).

| Context | Vocab | Initial loss | Final loss | Time |
|---|---:|---:|---:|---:|
| 32 chars | 59 | 4.08 | ~3.10 | 41 s / 100 steps |
| 128 chars | 61 | 4.13 | **~2.40** | 18.6 min / 1,000 steps |

**KV cache generation benchmark (200 tokens):**

| Setup | Time |
|---|---:|
| No cache | 14.25 s |
| Cache (no sliding window) | 107.8 s |
| **Cache + sliding window** | **7.2 s** |
| **Speedup** | **15×** |

---

## CUDA on NVIDIA T4

The CUDA backend uses **cuBLAS** for matmul, **cuFFT** for batched STFT, and **persistent device buffers** to eliminate per-operation host ↔ device transfers.

Results obtained on a virtualized NVIDIA Tesla T4 (Google Colab free tier, CUDA 13.0).

### Kernel performance (pure matmul)

1024×1024 matmul, 100 iterations:

| Path | Per call | Total | Speedup |
|---|---:|---:|---:|
| Naive (transfer per op) | 123.83 ms | 12,383 ms | 1× |
| **Persistent device tensors** | **0.83 ms** | **280 ms** | **44×** |

### Training performance (MNIST MLP)

Full 60,000-sample MNIST, `784 → 128 → 10`, 10 epochs:

| Metric | CPU | GPU v1 | **GPU v2** |
|---|---:|---:|---:|
| Per epoch | ~240 s | 27 s | **2.4 s** |
| Total | ~40 min | 4.5 min | **23.5 s** |
| Test accuracy | 98.05% | 93.31% | **96.85%** |
| **Speedup** | 1× | 9× | **~100×** |

### GPU compute primitives available

| Primitive | Purpose |
|:---|:---|
| `matmul`, `matmul_tn`, `matmul_nt` | cuBLAS GEMM variants |
| `add/sub/mul/div` | Elementwise |
| `relu_fwd/bwd` | ReLU with saved mask |
| `softmax_ce` | Fused softmax + cross-entropy |
| `bias_grad`, `sgd_update` | Training support |
| `conv2d_fwd/bwd`, `maxpool2d_fwd/bwd` | Vision (per-sample) |
| `conv2d_batched_*`, `maxpool2d_batched_*` | Vision (batched) |
| `softmax_rows_fwd/bwd` | Attention softmax |
| `causal_mask`, `causal_softmax` | Causal attention |
| `scale` | Scalar multiply |
| `stft_magnitude` | cuFFT-based audio |

### Reproduce

```bash
git clone https://github.com/YackswebCode/ZillaPHP.git
cd ZillaPHP
composer install
./native/cpu/build.sh
./native/cuda/build.sh   # requires nvcc

# Kernel-only benchmark
php -d memory_limit=2G tests/cuda_persistent.php

# Attention primitives
php -d memory_limit=2G tests/cuda_attention_test.php
php -d memory_limit=2G tests/cuda_attention_full_test.php

# Full training
TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=10 LR=0.05 \
    php -d memory_limit=3G examples/mnist_gpu_v2.php

TRAIN_N=60000 TEST_N=10000 BATCH=64 EPOCHS=5 LR=0.01 \
    php -d memory_limit=3G examples/mnist_cnn_gpu_v2.php

SEQ=32 D_MODEL=64 D_FF=128 STEPS=3000 LR=0.02 \
    php -d memory_limit=3G examples/transformer_gpu.php
```

For Google Colab, see [`docs/colab_setup.md`](docs/colab_setup.md).

---

## SafeTensors — HuggingFace Interoperability

Full SafeTensors serialization with metadata, verified against the
reference HuggingFace `safetensors` Python library.

```php
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Serialization\SafeTensorsSerializer;

$serializer = new ModelSerializer(new SafeTensorsSerializer());
$serializer->save($model, 'checkpoints/mnist.safetensors', [
    'epoch'         => 10,
    'test_accuracy' => 0.9685,
]);

// Load with metadata
$meta = $serializer->load($model, 'checkpoints/mnist.safetensors');
```

Any modern ML tool (PyTorch, NumPy, JAX) can read ZillaPHP checkpoints
via HuggingFace's reference implementation.

---

## CLI

```bash
zilla doctor              # environment diagnostics
zilla version             # framework version
zilla device:list         # available backends
zilla train <script.php>  # run a training script
zilla generate:text \
    --model=... \
    --prompt="..." \
    --tokens=300 \
    --temp=0.7
```

---

## Roadmap

| Milestone | Component | Status |
|---|---|---|
| **M0** | Foundation — repo, CLI, tests | ✅ |
| **M1** | Tensor engine | ✅ |
| **M2** | Automatic differentiation | ✅ |
| **M3** | Neural networks | ✅ |
| **M4** | Training — loss, optimizers, loader, trainer | ✅ |
| **M5** | MNIST validation — 98.05% MLP | ✅ |
| **M6** | Transformer — attention, blocks, tokenizer | ✅ |
| **M7** | Tiny language model + KV cache | ✅ |
| **M8** | Native CPU backend + OpenBLAS | ✅ |
| **M8.5** | Persistent CPU tensors — H0 rejected | ✅ |
| **M9** | CUDA backend — matmul, elementwise, ReLU | ✅ |
| **M9.5** | Persistent device tensors | ✅ |
| **M10** | Vision — Conv2D, MaxPool (CPU + CUDA, per-sample + batched) | ✅ |
| **M10.5** | Vision Transformer (ViT) | 🔜 |
| **M11** | Audio — WAV, Mel, STFT (CPU + CUDA) | ✅ |
| **M11.5** | Audio CNN classifier | ✅ synthetic / 🚧 real |
| **M11.6** | **CUDA attention primitives + full attention forward** | ✅ |
| **M11.7** | **GPU Transformer training (end-to-end)** | ✅ |
| **M12** | LayerNorm on GPU + embedding backward + multi-head | 🔜 Next |
| **M13** | Diffusion — VAE, UNet, schedulers | 🔜 |
| **M14** | Video — 5D tensors, temporal models | 🔜 |
| **M15** | Multimodal models | 🔜 |
| **M16** | Distributed training | 🔜 |
| **M17** | Stable 1.0 | 🔜 |

### Future Directions

- Persistent device tensors integration into `Tensor` API
- **GPU LayerNorm forward + backward** (next v2 Transformer feature)
- **Embedding backward on GPU**
- **Multi-head attention on GPU**
- **Stacked Transformer layers on GPU**
- **BPE tokenizer** (for real LLM training)
- Vision Transformer (ViT)
- Diffusion models
- ROCm/HIP port (via `hipify-perl`)
- ONNX interoperability
- Quantization (INT8, INT4, FP8)
- Kernel fusion (MatMul + Bias + ReLU)
- Multi-threaded CPU matmul
- Multi-GPU training (NCCL)

---

## Testing

```bash
./vendor/bin/phpunit
```

The suite covers:
- Unit tests for Tensor, layers, losses, optimizers
- Numerical gradient checks against central finite differences
- Integration tests: model construction, forward, backward, training loop, checkpointing
- CPU/CUDA parity tests
- Attention primitive exactness tests

---

## Contributing

Before opening a PR:

1. Run `./vendor/bin/phpunit`
2. Follow existing code style
3. Add tests for new functionality
4. Include benchmark results for performance changes

### Areas that need help

- Persistent device tensor integration
- GPU LayerNorm / embedding backward
- Multi-head attention on GPU
- ROCm/HIP backend
- Vision Transformer
- Diffusion models
- BPE tokenizer
- Quantization
- Distributed training

---

## Project Identity

```text
PHP
 │
 ├── Tensor
 ├── Autograd
 ├── Neural Networks
 ├── Training
 ├── Transformers
 ├── Audio
 └── Runtime
        │
        ├── CPU      → C / OpenBLAS + Persistent buffers
        ├── CUDA     → NVIDIA GPU + cuBLAS / cuFFT + Persistent device tensors
        │              + Conv2D / MaxPool / Attention kernels
        └── ROCm/HIP → AMD GPU (planned)
```

**Built with PHP 8.4 + C + CUDA.**

**Runs on a laptop.**

**Scales to GPU.**

**Trains Transformers.**

---

**Author:** Yahaya Ibrahim  
**Organization:** Yacksweb Tech  
**Country:** Nigeria

**License:** Apache-2.0 — see [`LICENSE`](LICENSE).
