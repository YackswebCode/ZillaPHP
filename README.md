# ZillaPHP

**A PHP-native framework for machine learning and artificial intelligence.**

Build, train, and run neural networks directly from PHP — with a pure-PHP API,
an extensible backend system, and native C acceleration through FFI.

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
| Neural network modules (`Linear`, `ReLU`, `Sequential`, `Embedding`, `LayerNorm`) | ✅ Working |
| Losses (`MSE`, `CrossEntropy`) | ✅ Working |
| Optimizers (`SGD`, `Adam`, `AdamW`) | ✅ Working |
| Batched training (`DataLoader`, `Trainer::fitLoader`) | ✅ Working |
| Model checkpointing (JSON, SafeTensors planned) | ✅ Working |
| Native C++ CPU backend via PHP FFI | ✅ Working |
| OpenBLAS integration (auto-detected) | ✅ Working |
| Real MNIST training (full 60,000 samples) | ✅ **98.05%** |
| Multi-head attention + Transformer blocks | ✅ Working |
| Character-level tokenizer | ✅ Working |
| Tiny language model on Shakespeare | ✅ Working |
| KV cache for fast generation | ✅ Working (3.5×) |
| CLI (`zilla doctor`, `zilla train`, `zilla generate:text`) | ✅ Working |
| Convolutional layers (Conv2D, MaxPool) | 🔜 In progress |
| CUDA backend | 🔜 Planned |
| ROCm backend | 🔜 Planned |
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
- **Native C kernels via FFI** — up to **160× faster matmul** than pure PHP
- **OpenBLAS integration** — `cblas_sgemm` for large matrices, auto-detected at build time
- **Transformer support** — multi-head attention, causal masking, positional embeddings
- **Language modeling** — character tokenizer, autoregressive sampler, KV cache
- **Extensible backend contract** — implement one interface to add a new device
- **CLI** — `zilla doctor`, `zilla train`, `zilla generate:text`, `zilla device:list`
- **Honest maturity** — experimental features are clearly marked

---

## Benchmarks

### Matmul (Acer TravelMate, 4-core Intel, AVX2, single-threaded OpenBLAS)

| Operation     | Pure PHP  | Native C | Speedup |
|---------------|-----------|----------|---------|
| matmul 64×64  | 44 ms     | 2.0 ms   | 22×     |
| matmul 128×128| 250 ms    | 8.3 ms   | 30×     |
| matmul 256×256| 1622 ms   | 27 ms    | 60×     |
| matmul 512×512| 16077 ms  | 103 ms   | **160×**|

> **Note on threading:** Single-threaded OpenBLAS is faster than multi-threaded
> for our matrix sizes on this machine. `Application::boot()` sets
> `OPENBLAS_NUM_THREADS=1` by default. Override with `ZILLA_OPENBLAS_THREADS=N`.

Run benchmarks yourself:

```bash
php -d memory_limit=2G benchmarks/matmul.php
```

### MNIST (full 60,000 samples, MLP 784→256→128→10)

| Metric | Value |
|--------|-------|
| Best test accuracy | **98.05%** |
| Epoch reached | 5 (early stop at 8) |
| Time per epoch | ~7 min |
| Total training time | **66.6 min** |

Reproduce:

```bash
php -d memory_limit=2G examples/mnist_full.php
```

### Shakespeare language model (character-level Transformer)

| Metric | Value |
|--------|-------|
| Model | dim=64, heads=4, layers=2, ~76k params |
| Vocab | 59 chars |
| Initial loss | ~4.08 (baseline `log(59)`) |
| Loss after 100 steps | **~3.10** |
| Generation (200 tokens, with KV cache) | **4.05 s** |
| Generation (200 tokens, without cache) | 14.25 s |
| **KV cache speedup** | **3.5×** |

---

## Requirements

- **PHP 8.4+** with `FFI` enabled
- **C compiler** (GCC or Clang)
- **OpenBLAS** (optional but recommended)
- **Composer**
- Linux / macOS (Windows untested)

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

### 3. Install OpenBLAS (recommended)

```bash
# Debian / Ubuntu / Linux Mint
sudo apt install -y libopenblas-dev
```

### 4. Build native kernels

```bash
./native/cpu/build.sh
```

The build script auto-detects OpenBLAS. Output:

```
OpenBLAS found via pkg-config
Built: /home/you/ZillaPHP/native/cpu/libzilla_cpu.so
```

If OpenBLAS is not installed, matmul falls back to a naive C loop. Still
works, just slower.

### 5. Verify

```bash
./bin/zilla doctor
```

Expected output:

```
ZillaPHP Doctor
--------------------------------------------
PHP version    : 8.4.x
ZillaPHP       : 0.1.0-dev
OS             : Linux
Architecture   : x86_64
CPU cores      : 4
Memory limit   : 2G
FFI enabled    : yes
Native lib     : present

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

echo $a->add($b)->data();     // [6, 8, 10, 12]
echo $a->matmul($b)->data();  // [19, 22, 43, 50]
echo $a->relu()->data();      // [1, 2, 3, 4]
```

### Automatic differentiation

```php
$x = Tensor::fromArray([2.0], requiresGrad: true);
$y = $x->mul($x);         // y = x²
$y->backward();
echo $x->grad()->item();  // 4.0
```

### Train a neural network

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

$optimizer = new Adam($model->parameters(), lr: 0.001);
$trainer   = new Trainer($model, $optimizer, new CrossEntropy());

$trainer->fit($samples, epochs: 10, onEpoch: fn($e, $loss) =>
    printf("epoch %d  loss %.4f\n", $e, $loss)
);
```

### Train a language model

```bash
# Download Shakespeare (1.1 MB, one time)
php scripts/download_shakespeare.php

# Train a tiny Transformer (~40 s for 100 steps)
TRAIN_CHARS=50000 SEQ_LEN=32 EPOCHS=1 STEPS_PER_EPOCH=100 \
  php -d memory_limit=2G examples/train_shakespeare.php

# Generate text from the checkpoint
./bin/zilla generate:text \
  --model=checkpoints/shakespeare.zilla.json \
  --prompt="ROMEO: " \
  --tokens=200 \
  --temp=0.8
```

Sample output after 100 training steps (character-by-character, still learning):

```
ROMEO: flt rkeice R
Skelaon Ayert
;v oeli
R
I ne Eauf n se llune the n  utsgrae ade me ngn dol dend n nl  uslet tsgdod n n udiir nre udef bde U
```

The model has learned spacing, capitalization, punctuation, and dialogue
structure — but not yet real English words. Train longer for coherent output.

### Train on real MNIST

```bash
# Download the dataset (~11 MB, one time)
php scripts/download_mnist.php

# Full 60,000-sample training (~66 min on 4-core CPU)
php -d memory_limit=2G examples/mnist_full.php
```

### Run the test suite

```bash
./vendor/bin/phpunit
```

Includes numerical gradient-checking against finite differences for every
differentiable operation.

---

## Architecture

ZillaPHP separates the developer-facing API from hardware execution:

```
┌──────────────────────────────────────┐
│         PHP Application              │
├──────────────────────────────────────┤
│  Tensor · Autograd · NN · Trainer    │   ← Pure PHP
├──────────────────────────────────────┤
│  Operator Dispatcher · Backend API   │   ← Pure PHP
├──────────────────────────────────────┤
│     CPU        │      CUDA           │   ← Native
│  (C via FFI)   │      (planned)      │
└──────────────────────────────────────┘
```

Key design rules:

1. **High-level PHP must not depend on a specific backend.**
   Model code is backend-agnostic; `Tensor::setBackend()` picks the runtime.

2. **Computationally intensive operations are delegated to native code.**
   Matmul, ReLU, elementwise ops, reductions — all run in compiled C.

3. **Correctness before optimization.**
   Every gradient function is tested against numerical finite differences.

4. **Reproducible benchmarks.**
   Performance claims come from `benchmarks/`, not from guesswork.

---

## CLI

```bash
zilla doctor                              # Diagnostic report
zilla version                             # Framework version
zilla device:list                         # Available compute devices
zilla train <example.php>                 # Run a training script
zilla generate:text \
  --model=<checkpoint.json> \
  --prompt="ROMEO: " \
  --tokens=200 \
  --temp=0.8                              # Generate text from a language model
```

---

## Project Structure

```
ZillaPHP/
├── bin/
│   └── zilla                     CLI entrypoint
├── src/
│   ├── Core/                     Application boot, contracts
│   ├── Tensor/                   Tensor, Shape, DType, Device
│   ├── Autograd/                 Graph, backward functions
│   ├── NN/                       Module, Parameter, layers
│   │   ├── Layers/               Linear, Embedding, LayerNorm, FeedForward
│   │   └── Activations/          ReLU
│   ├── Attention/                MultiHeadAttention
│   ├── Transformers/             TransformerBlock, TransformerLM, KVCache
│   ├── Tokenizers/               CharTokenizer
│   ├── Generators/               Sampler
│   ├── Loss/                     MSE, CrossEntropy
│   ├── Optim/                    SGD, Adam
│   ├── Data/                     Dataset, DataLoader, MnistDataset
│   ├── Training/                 Trainer, metrics
│   ├── Serialization/            Model checkpoints
│   ├── Runtime/                  Dispatcher, kernel registry
│   ├── Hardware/
│   │   └── CPU/                  CpuBackend, NativeCpuBackend (FFI)
│   └── CLI/                      Console commands
├── native/
│   └── cpu/                      C kernels + build script
├── tests/
│   ├── Unit/                     Component tests
│   ├── Numerical/                Gradient-checking vs finite differences
│   └── Integration/              End-to-end pipeline tests
├── examples/                     Runnable demos
├── benchmarks/                   Reproducible performance tests
├── scripts/                      Dataset downloaders
└── checkpoints/                  Saved model weights
```

---

## Roadmap

Following the [ZillaPHP paper](ZillaPHP-Paper.pdf) milestone order:

| Milestone | Component | Status |
|-----------|-----------|--------|
| **M0** | Foundation (repo, CLI, tests) | ✅ |
| **M1** | Tensor engine | ✅ |
| **M2** | Automatic differentiation | ✅ |
| **M3** | Neural networks | ✅ |
| **M4** | Training (loss, optim, loader, trainer) | ✅ |
| **M5** | MNIST validation (**98.05%**) | ✅ |
| **M6** | Transformer (attention, blocks, tokenizer) | ✅ |
| **M7** | Tiny language model + KV cache | ✅ |
| **M8** | Native CPU backend (OpenBLAS) | ✅ |
| **M9** | CUDA backend | 🔜 |
| **M10** | Vision (Conv2D, ViT) | 🚧 In progress |
| **M11** | Audio (waveform, spectrogram, STFT) | 🔜 |
| **M12** | Diffusion (VAE, UNet, schedulers) | 🔜 |
| **M13** | Video (5D tensors, temporal models) | 🔜 |
| **M14** | Multimodal | 🔜 |
| **M15** | Distributed training | 🔜 |
| **M16** | Stable 1.0 | 🔜 |

---

## Reproducible Benchmarks

Every performance claim is reproducible:

```bash
php -d memory_limit=2G benchmarks/matmul.php
```

If you publish results, please include:
- CPU model
- OS + PHP version
- Compiler version
- Backend (`cpu` vs `cpu+native`)
- OpenBLAS version and thread count
- Tensor shapes

---

## Testing

```bash
./vendor/bin/phpunit
```

Includes:

- **Unit tests** for tensor operations, shape inference, broadcasting
- **Numerical gradient tests** — every autograd function is compared against
  central finite differences with tolerances matched to float32 precision
- **Integration tests** — full training loops

---

## Documentation

- **Paper** — [`ZillaPHP-Paper.pdf`](ZillaPHP-Paper.pdf) — architecture, research, roadmap
- **Examples** — [`examples/`](examples/) — every feature has a runnable demo
- **Benchmarks** — [`benchmarks/`](benchmarks/) — performance measurement scripts

---

## Contributing

Contributions welcome. Before opening a pull request:

1. Run the full test suite: `./vendor/bin/phpunit`
2. Follow the existing code style (strict types, typed properties)
3. Add tests for new functionality
4. Keep public APIs documented

### Areas that especially need help

- Native CUDA backend
- Convolutional layers (Conv2D, MaxPool)
- Additional optimizers and losses
- Vision / audio preprocessing pipelines
- Documentation and examples

---

## License

**Apache-2.0** — see [LICENSE](LICENSE).

Chosen to support research use, commercial use, and a growing ecosystem of
native extensions.

---

## Acknowledgements

ZillaPHP draws architectural inspiration from PyTorch (imperative autograd),
TensorFlow (computation graphs, backend abstraction), and JAX (composable
transformations). It does not attempt to reproduce any of them — its goal is
a first-class PHP environment for machine learning.

---

**Built with PHP 8.4 + C. Runs on a laptop. Scales (eventually) to GPUs.**

*Author: Yahaya Ibrahim · Yacksweb Tech · Nigeria*
```
