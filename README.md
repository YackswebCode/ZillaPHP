# ZillaPHP

**A PHP-native framework for machine learning and artificial intelligence.**

Build, train, and run neural networks directly from PHP — with a PHP-first API, an extensible runtime/backend system, and native C/CUDA acceleration through FFI.

**Text · Audio · Image · Video**

---

## Status

> **⚠️ Experimental — under active development.**
>
> ZillaPHP is a research and engineering project, not a production-ready replacement for PyTorch or TensorFlow. APIs may change before 1.0.
>
> Performance claims are based on reproducible benchmarks. See [`benchmarks/`](benchmarks/).

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
| SafeTensors support | 🔜 Planned |
| Native C CPU backend through PHP FFI | ✅ Working |
| OpenBLAS integration with automatic detection | ✅ Working |
| Native Conv2D / MaxPool kernels through FFI | ✅ Working |
| Real MNIST training — MLP, full 60,000 samples | ✅ **98.05%** |
| Multi-head attention + Transformer blocks | ✅ Working |
| Character-level tokenizer | ✅ Working |
| Tiny character-level language model on Shakespeare | ✅ Working |
| KV cache for faster generation | ✅ Working (**15× measured speedup**) |
| CUDA backend — matmul, elementwise operations, ReLU | ✅ Working |
| CLI — `zilla doctor`, `zilla train`, `zilla generate:text` | ✅ Working |
| MNIST CNN full training | 🚧 Running |
| Vision Transformer (ViT) | 🔜 Planned |
| Audio pipeline — waveform, STFT | 🔜 Planned |
| Diffusion models | 🔜 Planned |
| Video models | 🔜 Planned |
| ROCm/HIP backend | 🔜 Planned |
| Multimodal models | 🔜 Planned |
| Distributed training | 🔜 Planned |

---

# Table of Contents

- [Why ZillaPHP Exists](#why-zillaphp-exists)
- [Design Philosophy](#design-philosophy)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
  - [CPU Matmul](#cpu-matmul)
  - [MNIST MLP](#mnist-mlp)
  - [MNIST CNN](#mnist-cnn)
  - [Shakespeare Language Model](#shakespeare-language-model)
  - [CUDA on NVIDIA T4](#cuda-on-nvidia-t4)
- [CLI](#cli)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Reproducible Benchmarks](#reproducible-benchmarks)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

# Why ZillaPHP Exists

Modern machine learning frameworks are concentrated around Python and C++.

Meanwhile, PHP powers a huge portion of the web — CMS platforms, APIs, financial systems, enterprise applications, SaaS products, and backend services.

When those applications need AI capabilities, developers often have to introduce another ecosystem:

- A Python microservice
- A REST API
- A separate inference server
- A language binding
- A second deployment environment

ZillaPHP proposes a different approach:

> **Make PHP the primary machine-learning development language while delegating computationally intensive operations to optimized native execution backends.**

The high-level API remains PHP-first, while hardware-specific execution is handled by dedicated backends.

The intended architecture is:

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

The developer writes PHP.

The runtime determines where numerical operations should execute.

---

# Design Philosophy

ZillaPHP is built around several principles.

## 1. PHP-first development

PHP should remain the primary developer-facing language.

Model definitions, tensors, training loops, optimizers, datasets, and application integration should be accessible directly from PHP.

```php
$model->forward($x);

$loss->backward();

$optimizer->step();
```

---

## 2. PHP should orchestrate, not perform every heavy computation

Pure PHP is useful for:

- API design
- Model composition
- Runtime orchestration
- Training control flow
- Data pipelines
- Configuration
- Application integration

But computationally intensive operations should be delegated to optimized native implementations.

Examples include:

- Matrix multiplication
- Convolution
- Reductions
- Elementwise operations
- Pooling
- GPU kernels

---

## 3. Separate execution backends

ZillaPHP does not rely on one technology to handle every hardware platform.

The execution layer is intentionally separated:

```text
                 ZillaPHP Runtime
                       │
             Operator Dispatcher
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   CPU Backend    CUDA Backend   ROCm/HIP Backend
        │              │              │
        ▼              ▼              ▼
   C / OpenBLAS     NVIDIA GPU       AMD GPU
```

### CPU

The CPU backend uses native C code, optionally accelerated with OpenBLAS.

### NVIDIA GPU

The CUDA backend provides NVIDIA GPU execution using CUDA kernels and can later integrate more deeply with CUDA libraries such as cuBLAS.

### AMD GPU

A future ROCm/HIP backend will provide AMD GPU support.

> **Important:** TypePHP or another PHP execution technology is not intended to be the CUDA or ROCm execution layer. CPU, CUDA, and ROCm/HIP are separate backend implementations.

---

## 4. Backend-independent model code

High-level model code should not need to know which hardware backend is being used.

For example:

```php
$x = Tensor::fromArray($data);

$output = $model->forward($x);
```

The runtime determines the appropriate execution backend.

---

## 5. Correctness before optimization

Every differentiable operation should be tested against numerical finite differences where practical.

Optimization comes after correctness.

---

## 6. Graceful fallback

When native acceleration is unavailable, ZillaPHP can fall back to pure PHP implementations where supported.

The goal is:

```text
Native backend available
        ↓
Use optimized execution

Native backend unavailable
        ↓
Use PHP fallback where supported
```

The expected difference is performance, not API-level correctness.

---

## 7. Reproducible performance

Performance claims should come from reproducible benchmark scripts rather than theoretical estimates.

Every published benchmark should include:

- Hardware
- Operating system
- PHP version
- Compiler version
- Backend
- OpenBLAS version/thread count
- Tensor dimensions
- Relevant runtime configuration

---

# Features

## Core Tensor Engine

- Tensor creation and manipulation
- Shape tracking
- Strides
- Dtypes
- Device abstraction
- Broadcasting
- Matrix multiplication
- Elementwise operations
- Reductions
- Activation functions

---

## Automatic Differentiation

- Reverse-mode autograd
- Dynamic computation graphs
- Gradient propagation
- Numerical gradient checking
- Finite-difference validation

---

## Neural Networks

Current modules include:

- `Linear`
- `ReLU`
- `Sequential`
- `Embedding`
- `LayerNorm`
- `Conv2D`
- `MaxPool2D`
- `Flatten`

---

## Training

- `Dataset`
- `DataLoader`
- `Trainer`
- Mini-batch training
- Loss functions
- Optimizers
- Checkpointing
- Training metrics

---

## Losses

- `MSE`
- `CrossEntropy`

---

## Optimizers

- `SGD`
- `Adam`

---

## Transformers

- Multi-head attention
- Causal masking
- Positional embeddings
- Transformer blocks
- Character-level tokenization
- Autoregressive generation
- KV cache
- Sliding-window KV cache

---

## Native CPU Acceleration

ZillaPHP can execute computationally intensive operations through native C libraries using PHP FFI.

Supported native operations currently include:

- Matrix multiplication
- Elementwise addition
- Elementwise subtraction
- Elementwise multiplication
- Elementwise division
- ReLU
- Conv2D
- MaxPool2D

OpenBLAS is automatically detected when available.

---

## CUDA Backend

The CUDA backend currently supports:

- Matrix multiplication
- Elementwise operations
- ReLU

The current implementation has been tested on an NVIDIA T4 in Google Colab.

Future CUDA work includes:

- Persistent device tensors
- More GPU kernels
- CUDA convolution
- Better memory management
- Reduced host/device transfers
- Kernel fusion
- Deeper CUDA library integration

---

## Extensible Backend Contract

ZillaPHP uses a backend abstraction so new execution devices can be added without rewriting the high-level framework.

The intended backend structure is:

```text
Backend Interface
      │
      ├── CPU Backend
      │     └── C / OpenBLAS
      │
      ├── CUDA Backend
      │     └── NVIDIA CUDA
      │
      └── ROCm/HIP Backend
            └── AMD GPU
```

---

## CLI

Available commands include:

```bash
zilla doctor
zilla version
zilla device:list
zilla train
zilla generate:text
```

---

# Architecture

ZillaPHP separates the developer-facing framework from hardware execution.

```text
┌──────────────────────────────────────────────────┐
│                 PHP Application                  │
├──────────────────────────────────────────────────┤
│        Tensor · Autograd · NN · Trainer          │
│                     Pure PHP                     │
├──────────────────────────────────────────────────┤
│        Runtime · Dispatcher · Backend API        │
│                     Pure PHP                     │
├──────────────────────┬───────────────────────────┤
│      CPU Backend     │       GPU Backends        │
│                      │                           │
│      Native C        │     CUDA / ROCm-HIP       │
│      OpenBLAS        │     NVIDIA / AMD          │
└──────────────────────┴───────────────────────────┘
```

### Runtime Flow

```text
Tensor Operation
      │
      ▼
Runtime
      │
      ▼
Operator Dispatcher
      │
      ▼
Backend Registry
      │
      ├── CPU
      ├── CUDA
      └── ROCm/HIP
             │
             ▼
       Native Kernel
```

### Architectural Rules

### Rule 1 — High-level PHP is backend-independent

Model code should not depend directly on CPU, CUDA, or ROCm implementations.

---

### Rule 2 — Heavy computation belongs in native execution layers

Operations such as:

- Matmul
- Conv2D
- MaxPool
- Reductions
- Elementwise operations

should execute in optimized native code where available.

---

### Rule 3 — Runtime selects the backend

The runtime and dispatcher are responsible for selecting the appropriate backend for the requested operation.

---

### Rule 4 — Avoid unnecessary data movement

A major future optimization is persistent device tensors.

Instead of:

```text
PHP
 ↓
CPU memory
 ↓
GPU
 ↓
CPU memory
 ↓
GPU
```

the framework should eventually support:

```text
PHP Runtime
     ↓
GPU-resident Tensor
     ↓
GPU Operation
     ↓
GPU Operation
     ↓
GPU Operation
```

This is particularly important for reducing repeated host/device transfer overhead.

---

### Rule 5 — Correctness before performance

Backend optimizations must preserve numerical correctness.

---

# Requirements

## Minimum

- **PHP 8.4+**
- **FFI enabled**
- **Composer**
- **C compiler** — GCC or Clang
- Linux or macOS

Windows is currently untested.

---

## Recommended for CPU acceleration

- OpenBLAS
- GCC or Clang
- AVX2-capable CPU where available

---

## Required for CUDA acceleration

- NVIDIA GPU
- CUDA Toolkit
- `nvcc`
- PHP FFI
- Linux environment recommended

---

## Future AMD GPU support

ROCm/HIP support is planned.

The ROCm backend will be developed separately from the CPU and CUDA backends.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/YackswebCode/ZillaPHP.git
cd ZillaPHP
```

---

## 2. Install PHP dependencies

```bash
composer install
```

---

## 3. Install OpenBLAS

On Debian, Ubuntu, or Linux Mint:

```bash
sudo apt install -y libopenblas-dev
```

OpenBLAS is recommended for CPU matrix operations.

---

## 4. Build the native CPU backend

```bash
./native/cpu/build.sh
```

The build script automatically detects OpenBLAS when available.

Example output:

```text
OpenBLAS found via pkg-config
Built: /home/you/ZillaPHP/native/cpu/libzilla_cpu.so
```

If OpenBLAS is not installed, the CPU backend can fall back to a native C implementation using a naive matrix multiplication loop.

It will still work, but performance will be lower.

---

## 5. Build the CUDA backend

This step is optional.

Only perform it when an NVIDIA GPU and CUDA Toolkit are available.

```bash
./native/cuda/build.sh
```

The CUDA build requires:

```bash
nvcc
```

On Google Colab, the CUDA environment can be prepared according to the instructions in the project documentation.

---

## 6. Verify the installation

Run:

```bash
./bin/zilla doctor
```

Example:

```text
ZillaPHP Doctor
--------------------------------------------
PHP version    : 8.4.x
ZillaPHP       : 0.1.0-dev
OS             : Linux
Architecture   : x86_64
CPU cores      : 4
Memory limit   : 2G
FFI enabled    : yes
CPU lib        : present
CUDA lib       : not built
CUDA available : no

Recommended backend: CPU
```

On a machine with a correctly configured CUDA backend, the CUDA fields should report that CUDA is available.

---

# Quick Start

## Tensor Basics

```php
use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

Application::boot();

$a = Tensor::fromArray([
    [1.0, 2.0],
    [3.0, 4.0],
]);

$b = Tensor::fromArray([
    [5.0, 6.0],
    [7.0, 8.0],
]);

echo $a->add($b)->data();
// [6, 8, 10, 12]

echo $a->matmul($b)->data();
// [19, 22, 43, 50]

echo $a->relu()->data();
// [1, 2, 3, 4]
```

---

# Automatic Differentiation

```php
$x = Tensor::fromArray(
    [2.0],
    requiresGrad: true
);

$y = $x->mul($x); // y = x²

$y->backward();

echo $x->grad()->item();
// 4.0
```

---

# Train a Neural Network

```php
use ZillaPHP\NN\Sequential;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Training\Trainer;

$model = new Sequential([
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10),
]);

$optimizer = new Adam(
    $model->parameters(),
    lr: 0.001
);

$trainer = new Trainer(
    $model,
    $optimizer,
    new CrossEntropy()
);

$trainer->fit(
    $samples,
    epochs: 10,
    onEpoch: fn($e, $loss) =>
        printf(
            "epoch %d  loss %.4f\n",
            $e,
            $loss
        )
);
```

---

# Train a Convolutional Network

```php
use ZillaPHP\NN\Sequential;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Layers\Conv2D;
use ZillaPHP\NN\Layers\MaxPool2D;
use ZillaPHP\NN\Layers\Flatten;
use ZillaPHP\NN\Activations\ReLU;

$model = new Sequential([
    new Conv2D(
        1,
        8,
        kernelSize: 3,
        stride: 1,
        padding: 1
    ),

    new ReLU(),

    new MaxPool2D(2),

    new Conv2D(
        8,
        16,
        kernelSize: 3,
        stride: 1,
        padding: 1
    ),

    new ReLU(),

    new MaxPool2D(2),

    new Flatten(),

    new Linear(16 * 7 * 7, 10),
]);

// Input tensor:
// [C, H, W] = [1, 28, 28]
```

---

# Train a Language Model

Download the Shakespeare dataset:

```bash
php scripts/download_shakespeare.php
```

Train a tiny Transformer:

```bash
TRAIN_CHARS=100000 \
SEQ_LEN=128 \
EPOCHS=2 \
STEPS_PER_EPOCH=500 \
php -d memory_limit=2G examples/train_shakespeare.php
```

Generate text:

```bash
./bin/zilla generate:text \
  --model=checkpoints/shakespeare.zilla.json \
  --prompt="ROMEO: " \
  --tokens=300 \
  --temp=0.7
```

Example early-stage output:

```text
he ther the s rfinthee the He mave se.
INUS:

He cod thabrer ale thon harmyot thistis eangr'l t pe th t thico y
```

At this stage, the model has learned some:

- Spacing
- Capitalization
- Punctuation
- Dialogue structure
- Common English character patterns

It has not yet learned coherent words or high-quality language generation.

---

# Train on Real MNIST

Download MNIST:

```bash
php scripts/download_mnist.php
```

Train the MLP:

```bash
php -d memory_limit=2G examples/mnist_full.php
```

The current full MNIST MLP benchmark reaches:

```text
98.05% test accuracy
```

Train the CNN:

```bash
php -d memory_limit=2G examples/mnist_cnn.php
```

---

# Run the Test Suite

```bash
./vendor/bin/phpunit
```

The test suite includes numerical gradient checking against finite differences for differentiable operations such as:

- Tensor operations
- Conv2D
- MaxPool2D
- LayerNorm
- Softmax
- Other autograd functions

---

# Benchmarks

All benchmark results should be interpreted in the context of the hardware and runtime configuration used.

---

## CPU Matmul

### Acer TravelMate — 4-core Intel CPU, AVX2, single-threaded OpenBLAS

| Operation | Pure PHP | Native C | Speedup |
|---|---:|---:|---:|
| matmul 64×64 | 57 ms | 1.8 ms | **32×** |
| matmul 128×128 | 181 ms | 5.6 ms | **33×** |
| matmul 256×256 | 1706 ms | 36 ms | **47×** |
| matmul 784×128 | 1078 ms | 32 ms | **33×** |
| **matmul 512×512** | **25596 ms** | **122 ms** | **209×** |

### OpenBLAS Threading

For the tested matrix sizes on this machine, single-threaded OpenBLAS was faster than multi-threaded execution.

`Application::boot()` therefore sets:

```text
OPENBLAS_NUM_THREADS=1
```

by default.

Override it with:

```bash
ZILLA_OPENBLAS_THREADS=N
```

For example:

```bash
ZILLA_OPENBLAS_THREADS=4 \
php -d memory_limit=2G benchmarks/matmul.php
```

Run the benchmark yourself:

```bash
php -d memory_limit=2G benchmarks/matmul.php
```

---

# MNIST MLP

The current MLP uses the full 60,000-sample MNIST training set.

| Metric | Value |
|---|---:|
| Best test accuracy | **98.05%** |
| Best epoch | 5 |
| Early stopping | 8 |
| Time per epoch | ~7 min |
| Total training time | **66.6 min** |

Reproduce:

```bash
php -d memory_limit=2G examples/mnist_full.php
```

---

# MNIST CNN

The CNN uses native C kernels for convolution and pooling.

| Metric | Value |
|---|---:|
| Parameters | 9,098 |
| Native convolution speedup | **3.2×** over pure PHP |
| 200-sample proof of concept | 78% accuracy in 16.5 s |
| Full 60,000-sample run | 🚧 In progress |

Reproduce the proof of concept:

```bash
TRAIN_N=200 \
TEST_N=100 \
EPOCHS=2 \
php -d memory_limit=2G examples/mnist_cnn.php
```

---

# Shakespeare Language Model

The current language-model experiment uses a character-level Transformer.

| Context | Model | Vocab | Initial Loss | Final Loss | Training Time |
|---|---|---:|---:|---:|---:|
| 32 chars | dim=64, heads=4, layers=2 (~76k params) | 59 | 4.08 | ~3.10 | 41 s / 100 steps |
| **128 chars** | dim=64, heads=4, layers=2 (~76k params) | 61 | 4.13 | **~2.40** | **18.6 min / 1,000 steps** |

---

## KV Cache Generation Benchmark

Generation benchmark using 200 tokens:

| Setup | Time |
|---|---:|
| No cache | 14.25 s |
| Cache without sliding window | 107.8 s |
| **Cache + sliding window** | **7.2 s** |
| **Measured speedup** | **15×** |

The benchmark demonstrates the importance of controlling KV-cache growth during autoregressive generation.

---

## Reproduce

Train:

```bash
TRAIN_CHARS=100000 \
SEQ_LEN=128 \
EPOCHS=2 \
STEPS_PER_EPOCH=500 \
php -d memory_limit=2G examples/train_shakespeare.php
```

Generate:

```bash
./bin/zilla generate:text \
  --model=checkpoints/shakespeare.zilla.json \
  --prompt="ROMEO: " \
  --tokens=300 \
  --temp=0.7
```

---

# CUDA on NVIDIA T4

The CUDA backend currently includes a tiled GEMM implementation with pinned host memory and CUDA streams.

The following results were obtained on a virtualized NVIDIA T4 environment in Google Colab.

| Matrix Size | CPU + Native | CUDA T4 | Speedup |
|---|---:|---:|---:|
| 128×128 | 2.04 ms | 1.88 ms | 1.08× |
| 256×256 | 10.33 ms | 9.91 ms | 1.04× |
| 512×512 | 44.03 ms | 43.23 ms | 1.02× |
| **1024×1024** | **338.22 ms** | **199.55 ms** | **1.69×** |

### Interpretation

The measured Colab environment shows substantial launch and virtualization overhead for small and medium operations.

Therefore, these results should **not** be interpreted as a general measurement of bare-metal NVIDIA GPU performance.

The results motivate a major ZillaPHP optimization:

> **Persistent device tensors.**

Instead of repeatedly transferring tensors between CPU and GPU for every operation, future versions should keep tensors resident on the GPU across multiple operations whenever possible.

This should reduce:

- Host/device transfer overhead
- Repeated allocation
- Kernel-launch-related overhead
- Synchronization overhead

### Important Benchmarking Principle

ZillaPHP will not claim a fixed GPU speedup without measuring it on the target hardware.

Future bare-metal NVIDIA benchmarks should report the actual measured results.

---

## Reproduce CUDA Benchmark

On a CUDA-enabled environment:

```bash
git clone https://github.com/YackswebCode/ZillaPHP.git
cd ZillaPHP

./native/cuda/build.sh

php -d memory_limit=2G benchmarks/cuda_matmul.php
```

For Google Colab, see the CUDA setup documentation:

```text
docs/colab_setup.md
```

---

# CLI

The ZillaPHP CLI provides common framework operations.

## Doctor

```bash
zilla doctor
```

Displays:

- PHP version
- ZillaPHP version
- Operating system
- Architecture
- CPU cores
- Memory limit
- FFI status
- CPU backend status
- CUDA backend status
- Recommended backend

---

## Version

```bash
zilla version
```

---

## Device List

```bash
zilla device:list
```

---

## Run a Training Script

```bash
zilla train <example.php>
```

---

## Generate Text

```bash
zilla generate:text \
  --model=<checkpoint.json> \
  --prompt="ROMEO: " \
  --tokens=300 \
  --temp=0.7 \
  --seed=42
```

---

# Project Structure

```text
ZillaPHP/
│
├── bin/
│   └── zilla
│
├── src/
│   │
│   ├── Core/
│   │   └── Application boot, contracts
│   │
│   ├── Tensor/
│   │   └── Tensor, Shape, DType, Device
│   │
│   ├── Autograd/
│   │   └── Graph, backward functions
│   │
│   ├── NN/
│   │   ├── Module, Parameter, layers
│   │   ├── Layers/
│   │   │   ├── Linear
│   │   │   ├── Embedding
│   │   │   ├── LayerNorm
│   │   │   ├── FeedForward
│   │   │   ├── Conv2D
│   │   │   ├── MaxPool2D
│   │   │   └── Flatten
│   │   └── Activations/
│   │       └── ReLU
│   │
│   ├── Attention/
│   │   └── MultiHeadAttention
│   │
│   ├── Transformers/
│   │   ├── TransformerBlock
│   │   ├── TransformerLM
│   │   └── KVCache
│   │
│   ├── Tokenizers/
│   │   └── CharTokenizer
│   │
│   ├── Generators/
│   │   └── Sampler
│   │
│   ├── Loss/
│   │   ├── MSE
│   │   └── CrossEntropy
│   │
│   ├── Optim/
│   │   ├── SGD
│   │   └── Adam
│   │
│   ├── Data/
│   │   ├── Dataset
│   │   ├── DataLoader
│   │   └── MnistDataset
│   │
│   ├── Training/
│   │   ├── Trainer
│   │   └── metrics
│   │
│   ├── Serialization/
│   │   └── Model checkpoints
│   │
│   ├── Runtime/
│   │   ├── Dispatcher
│   │   ├── Backend registry
│   │   └── Execution management
│   │
│   ├── Hardware/
│   │   ├── CPU/
│   │   │   ├── CpuBackend
│   │   │   └── NativeCpuBackend
│   │   │
│   │   ├── CUDA/
│   │   │   └── CudaBackend
│   │   │
│   │   └── ROCm/
│   │       └── Future ROCm/HIP backend
│   │
│   └── CLI/
│       └── Console commands
│
├── native/
│   │
│   ├── cpu/
│   │   ├── build.sh
│   │   └── zilla_cpu.c
│   │       ├── matmul
│   │       ├── elementwise
│   │       ├── relu
│   │       ├── conv2d
│   │       └── maxpool
│   │
│   ├── cuda/
│   │   ├── build.sh
│   │   ├── matmul.cu
│   │   └── elementwise.cu
│   │
│   └── rocm/
│       └── Planned
│
├── tests/
│   ├── Unit/
│   │   └── Component tests
│   ├── Numerical/
│   │   └── Gradient checking
│   └── Integration/
│       └── End-to-end pipeline tests
│
├── examples/
│   └── Runnable demonstrations
│
├── benchmarks/
│   └── Reproducible performance tests
│
├── scripts/
│   └── Dataset downloaders
│
├── checkpoints/
│   └── Saved model weights
│
├── docs/
│   └── Documentation
│
└── ZillaPHP-Paper.pdf
```

---

# Roadmap

ZillaPHP development follows the milestone order described in the project paper.

| Milestone | Component | Status |
|---|---|---|
| **M0** | Foundation — repository, CLI, tests | ✅ |
| **M1** | Tensor engine | ✅ |
| **M2** | Automatic differentiation | ✅ |
| **M3** | Neural networks | ✅ |
| **M4** | Training — loss, optimizers, loader, trainer | ✅ |
| **M5** | MNIST validation — 98.05% MLP | ✅ |
| **M6** | Transformer — attention, blocks, tokenizer | ✅ |
| **M7** | Tiny language model + KV cache | ✅ |
| **M8** | Native CPU backend + OpenBLAS | ✅ |
| **M9** | CUDA backend — matmul, elementwise, ReLU | ✅ |
| **M10** | Vision — Conv2D, MaxPool | ✅ |
| **M10.5** | Vision Transformer (ViT) | 🔜 |
| **M11** | Audio — waveform, spectrogram, STFT | 🔜 |
| **M12** | Diffusion — VAE, UNet, schedulers | 🔜 |
| **M13** | Video — 5D tensors, temporal models | 🔜 |
| **M14** | Multimodal models | 🔜 |
| **M15** | Distributed training | 🔜 |
| **M16** | Stable 1.0 | 🔜 |

---

# Future Directions

## Persistent Device Tensors

Keep tensors resident on the GPU across multiple operations.

Goal:

```text
Host → GPU
         │
         ├── Operation 1
         ├── Operation 2
         ├── Operation 3
         └── Operation 4
```

rather than repeatedly moving data:

```text
Host → GPU → Host → GPU → Host → GPU
```

---

## ROCm/HIP Backend

Add AMD GPU support through a dedicated ROCm/HIP backend.

The intended architecture is:

```text
ZillaPHP Runtime
      │
      ├── CPU Backend
      │     └── C / OpenBLAS
      │
      ├── CUDA Backend
      │     └── NVIDIA GPU
      │
      └── ROCm/HIP Backend
            └── AMD GPU
```

---

## SafeTensors

Add SafeTensors model-weight export/import for interoperability with the wider machine-learning ecosystem.

---

## Quantization

Potential future inference paths:

- INT8
- FP16
- Other reduced-precision formats

---

## Kernel Fusion

Combine multiple operations into a single native kernel.

Example:

```text
MatMul
  ↓
Bias
  ↓
ReLU
```

could eventually become:

```text
Fused MatMul + Bias + ReLU Kernel
```

This can reduce intermediate memory operations and kernel-launch overhead.

---

# Reproducible Benchmarks

Every performance claim should be reproducible.

## CPU

```bash
php -d memory_limit=2G benchmarks/matmul.php
```

## CUDA

```bash
php -d memory_limit=2G benchmarks/cuda_matmul.php
```

When publishing results, include:

- CPU model
- GPU model, if applicable
- Operating system
- PHP version
- Compiler version
- `gcc --version`
- `nvcc --version`
- Backend
- OpenBLAS version
- OpenBLAS thread count
- Tensor shapes
- Relevant environment variables

Example:

```text
CPU:
Intel CPU model

GPU:
NVIDIA T4

OS:
Linux

PHP:
8.4.x

Compiler:
GCC x.x

CUDA:
CUDA x.x

Backend:
cuda

OpenBLAS:
x.x

Threads:
1

Tensor:
1024 × 1024
```

---

# Testing

Run the complete test suite:

```bash
./vendor/bin/phpunit
```

The test suite includes:

### Unit Tests

Tests for:

- Tensor operations
- Shape inference
- Broadcasting
- Layers
- Optimizers
- Loss functions
- Runtime components

### Numerical Gradient Tests

Differentiable operations are compared against central finite differences.

Current coverage includes operations such as:

- `conv2d`
- `maxpool2d`
- `layernorm`
- `softmax`
- Tensor arithmetic
- Other autograd operations

The purpose is to ensure that backend optimizations do not compromise numerical correctness.

### Integration Tests

End-to-end tests cover:

- Model construction
- Forward passes
- Backward passes
- Optimization
- Training loops
- Checkpointing

---

# Documentation

## Project Paper

[`ZillaPHP-Paper.pdf`](ZillaPHP-Paper.pdf)

The paper documents:

- Architecture
- Design motivation
- Runtime model
- Backend system
- Research direction
- Benchmarks
- Roadmap

---

## Examples

[`examples/`](examples/)

Runnable demonstrations covering current framework capabilities.

---

## Benchmarks

[`benchmarks/`](benchmarks/)

Reproducible performance measurement scripts.

---

# Contributing

Contributions are welcome.

Before opening a pull request:

1. Run the full test suite:

   ```bash
   ./vendor/bin/phpunit
   ```

2. Follow the existing code style.
3. Use strict types where appropriate.
4. Use typed properties.
5. Add tests for new functionality.
6. Keep public APIs documented.
7. Include benchmark results when making performance-related changes.
8. Clearly identify experimental functionality.

---

# Areas That Need Help

The following areas are especially important for future development:

- Persistent device tensors
- CUDA Conv2D kernels
- CUDA memory management
- ROCm/HIP backend
- Vision Transformer
- Audio preprocessing
- WAV loading
- STFT
- Additional optimizers
- Additional loss functions
- SafeTensors support
- Quantization
- Kernel fusion
- Distributed training
- Documentation
- Examples
- Benchmarking across different hardware

---

# Research Direction

ZillaPHP is not intended to simply reproduce existing Python machine-learning frameworks.

The project explores the question:

> **Can PHP provide a practical, extensible machine-learning development environment while delegating computational workloads to optimized native hardware backends?**

The research direction focuses on:

```text
PHP Developer Experience
          +
Backend Abstraction
          +
Native CPU Execution
          +
CUDA GPU Execution
          +
Future ROCm/HIP Execution
          +
Efficient Runtime Dispatch
          +
Reduced Data Movement
```

The project will use measurements and reproducible experiments rather than assuming that PHP can match native frameworks across every workload.

---

# License

**Apache-2.0** — see [`LICENSE`](LICENSE).

The Apache-2.0 license is intended to support:

- Research use
- Commercial use
- Open-source development
- Native extensions
- Ecosystem development

---

# Acknowledgements

ZillaPHP draws architectural inspiration from several machine-learning ecosystems:

- **PyTorch** — imperative programming and automatic differentiation
- **TensorFlow** — computation graphs and backend abstraction
- **JAX** — composable transformations

ZillaPHP does not attempt to reproduce any of these frameworks.

Its goal is to provide a **first-class PHP environment for machine learning and artificial intelligence**.

---

# Project Identity

**ZillaPHP**

A PHP-first machine-learning framework with native execution backends.

```text
PHP
 │
 ├── Tensor
 ├── Autograd
 ├── Neural Networks
 ├── Training
 ├── Transformers
 └── Runtime
        │
        ├── CPU → C / OpenBLAS
        ├── CUDA → NVIDIA GPU
        └── ROCm/HIP → AMD GPU
```

---

**Built with PHP 8.4 + C + CUDA.**

**Runs on a laptop.**

**Designed to scale to GPU execution.**

---

**Author:** Yahaya Ibrahim  
**Organization:** Yacksweb Tech  
**Country:** Nigeria