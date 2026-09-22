# Two Things — Next Steps + Session Handoff Doc

---

## Part 1 — What's Next

You're at a natural checkpoint. Here's the prioritized list.

### 🔥 Immediate (today, if you have energy)

**Fix the sampler** — 30 min
- `zilla generate:text --tokens=300` outputs only ~35 chars
- Bug is in `src/Generators/Sampler.php`
- Likely: newline character triggering early stop, or off-by-one in loop
- Small fix, real user-visible improvement

### 🎯 High-Value (next 2–4 days)

**CUDA Conv2D + MaxPool kernels** — 2–3 days
- Unlocks GPU training for CNN and audio models
- Same pattern as `training_ops.cu` — you already have the template
- Gives you: MNIST CNN on GPU at ~95%+ in seconds

**Update the paper** — 1 day
- Add the CPU persistent tensor results (H0 rejection)
- Add the GPU results (H1 confirmation)
- Update §22.7 with the new measurements
- This is the citable research output

### 🚀 Bigger Moves (next 2–4 weeks)

**Integrate persistent tensors into `Tensor`** — 2–3 days
- `$a->matmul($b)` auto-stays on device if inputs are device-resident
- Cleaner user API, less boilerplate
- Requires reworking `Tensor.php` — careful work

**Vision Transformer (ViT)** — 3–5 days
- M10.5 milestone
- Patch embedding, positional encoding, transformer blocks
- Reuses your existing attention code

**Audio CNN classifier** — 1–2 days
- Uses the CUDA Conv2D kernels once they exist
- Speech Commands dataset (or synthetic)
- End-to-end GPU audio pipeline

### 📋 Recommended Order

```
1. Fix sampler                    (30 min)  ← today
2. Update paper with results      (1 day)   ← preserves the win
3. CUDA Conv2D + MaxPool          (2-3 days)
4. Audio CNN on GPU               (1-2 days)
5. Integrate persistent into Tensor (2-3 days)
6. Vision Transformer             (3-5 days)
7. Diffusion models               (1-2 weeks)
```

**But first**: commit and push the README update. Then rest.

---

## Part 2 — Session Handoff Document

Save this as `docs/SESSION_HANDOFF.md` in your repo. When you start a fresh chat, paste this entire document as the first message.

```markdown
# ZillaPHP — Session Handoff

This document summarizes the project state so a new session can continue
without re-discovering context.

**Last updated:** 2026-09-22
**Author:** Yahaya Ibrahim (Yacksweb Tech, Nigeria)
**Repo:** https://github.com/YackswebCode/ZillaPHP

---

## What ZillaPHP Is

A PHP-native framework for machine learning and AI. The goal is not to
replace PyTorch, but to let PHP developers build, train, and run models
without leaving PHP.

**Architecture:**

```
PHP Application
      │
      ▼
Tensor / Neural Network API   (PHP)
      │
      ▼
Runtime / Dispatcher          (PHP)
      │
      ▼
Backend Selection
      │
      ├───────────────┬──────────────────┐
      ▼               ▼                  ▼
   CPU Backend     CUDA Backend      ROCm/HIP
   C + OpenBLAS    cuBLAS + cuFFT    (planned)
   + persistent    + persistent
     buffers         device tensors
```

**Key design rules:**
1. PHP-first API, native execution
2. Backend-independent model code
3. Persistent buffers — no per-op array conversion
4. Correctness before performance

---

## Current State (as of 2026-09-22)

### ✅ Working

| Component | Status |
|---|---|
| Tensor engine + autograd | ✅ |
| NN modules (Linear, ReLU, Conv2D, MaxPool2D, LayerNorm, Embedding, Flatten, Sequential) | ✅ |
| Losses (MSE, CrossEntropy) | ✅ |
| Optimizers (SGD, Adam) | ✅ |
| Data (Dataset, DataLoader, MnistDataset) | ✅ |
| Trainer | ✅ |
| Checkpointing (JSON) | ✅ |
| Transformer (attention, blocks, KV cache) | ✅ |
| Char-level tokenizer | ✅ |
| Audio (WAV, Hann, STFT, Mel, log-mel) | ✅ |
| Native CPU backend (C + OpenBLAS) | ✅ |
| CUDA backend (cuBLAS matmul, cuFFT STFT) | ✅ |
| **Persistent CPU tensors** | ✅ |
| **Persistent device tensors (GPU)** | ✅ |
| CLI (`zilla doctor`, `zilla train`, `zilla generate:text`) | ✅ |
| GPU-native MNIST training | ✅ 96.85% in 23.5 s |
| CPU-native MNIST training (persistent) | ✅ 96.86% in 52 s |

### 🚧 In Progress / Known Bugs

| Item | Status |
|---|---|
| MNIST CNN full training | 🚧 running |
| `zilla generate:text --tokens=N` | 🐛 emits only ~35 chars, not N |
| CUDA Conv2D / MaxPool kernels | 🔜 next up |
| Vision Transformer (ViT) | 🔜 planned |
| Diffusion models | 🔜 planned |

### 🔜 Planned

- ROCm/HIP backend (via hipify-perl)
- SafeTensors import/export
- ONNX interop
- Quantization (INT8, INT4, FP8)
- Kernel fusion
- Multi-GPU training
- Multimodal models
- Video models

---

## Measured Results (citable)

### MNIST MLP training (60,000 samples, 10 epochs)

| Path | Per epoch | Total | Test acc |
|---|---:|---:|---:|
| Old CPU (naive transfers) | ~400 s | ~66 min | 98.05% |
| **New CPU (persistent buffers)** | **5.2 s** | **52 s** | **96.86%** |
| GPU v1 (naive transfers, T4) | 27 s | 4.5 min | 93.31% |
| **GPU v2 (persistent, T4)** | **2.4 s** | **23.5 s** | **96.85%** |

**CPU speedup: ~77×**  (H0 rejected)
**GPU speedup: ~100×** (H1 confirmed)

### Kernel-only matmul (1024×1024, 100 iters, T4)

| Path | Per call | Total |
|---|---:|---:|
| Naive (transfer per op) | 123.83 ms | 12,383 ms |
| **Persistent device tensors** | **0.83 ms** | **280 ms** |
| **Speedup** | | **44×** |

### Audio pipeline (1-sec clip, CPU)

| Stage | Time |
|---|---:|
| WAV load | ~5 ms |
| Hann window | 0.17 ms |
| STFT (512-FFT, 160-hop) | 9.2 ms |
| Mel + log | ~10 ms |
| **Total** | **~25 ms** (40× real-time) |

### Shakespeare Transformer (76k params)

| Metric | Value |
|---|---|
| Loss, 1000 steps | 3.5 → 2.53 |
| Training time (CPU) | 22.4 min |

---

## Repository Layout

```
ZillaPHP/
├── bin/zilla              # CLI entrypoint
├── src/
│   ├── Core/              # Application, contracts
│   ├── Tensor/            # Tensor, Shape, DType, Device
│   ├── Autograd/          # gradient graph
│   ├── NN/                # layers, activations, modules
│   ├── Attention/         # multi-head attention
│   ├── Transformers/      # TransformerLM, TransformerBlock, KVCache
│   ├── Tokenizers/        # CharTokenizer
│   ├── Generators/        # Sampler (text generation)
│   ├── Loss/              # MSE, CrossEntropy
│   ├── Optim/             # SGD, Adam
│   ├── Data/              # Dataset, DataLoader, MnistDataset
│   ├── Training/          # Trainer, metrics
│   ├── Serialization/     # ModelSerializer
│   ├── Audio/             # Wav, MelFilterbank, MelSpectrogram
│   ├── Hardware/
│   │   ├── CPU/           # CpuBackend, NativeCpuBackend
│   │   └── CUDA/          # CudaBackend
│   └── CLI/               # Application (console)
├── native/
│   ├── cpu/               # zilla_cpu.c, audio.c, buffer_registry.c, training_ops.c
│   └── cuda/              # matmul.cu, elementwise.cu, audio.cu,
│                          # buffer_registry.cu, device_ops.cu, training_ops.cu
├── examples/              # runnable demonstrations
├── benchmarks/            # reproducible benchmarks
├── tests/                 # unit, numerical, integration
├── scripts/               # dataset downloaders
├── docs/                  # documentation
├── checkpoints/           # saved models (gitignored)
└── data/                  # datasets (gitignored)
```

---

## Environment / Build

**Minimum:**
- PHP 8.4+ with FFI enabled
- Composer
- GCC or Clang
- Linux (macOS untested but likely works)

**Recommended:**
- OpenBLAS (`sudo apt install libopenblas-dev`)

**For CUDA:**
- NVIDIA GPU
- CUDA Toolkit 12.x (`nvcc`)
- Build on Colab for free testing

**Build commands:**

```bash
composer install
./native/cpu/build.sh
./native/cuda/build.sh     # only if nvcc available
```

**Verify:**
```bash
./bin/zilla doctor
```

Should report `cpu+native`, `FFI enabled: yes`, and `CUDA lib: present` (on CUDA machines).

---

## Development Workflow

**Laptop (VS Code) — source of truth:**
```bash
cd ~/ZillaPHP
# edit files
git add .
git commit -m "..."
git pull --rebase origin main    # sync with Colab pushes
git push origin main
```

**Colab — GPU testing:**
```bash
cd /content
git clone https://github.com/YackswebCode/ZillaPHP.git
cd ZillaPHP
# install PHP (one-time per session)
# then:
./native/cpu/build.sh
./native/cuda/build.sh
php -d memory_limit=3G examples/mnist_gpu_v2.php
```

**Important:** Colab wipes `/content` between sessions. Every fresh session needs:
1. PHP install (~2 min)
2. Composer install
3. Native library builds
4. MNIST download (`php scripts/download_mnist.php`)

---

## Key APIs (current)

### Persistent CPU tensors

```php
$backend = \ZillaPHP\Tensor\Tensor::backend();   // NativeCpuBackend

$wId  = $backend->bufferAlloc(784 * 128);   // allocate
$backend->bufferUpload($wId, $wData);        // host → buffer

$backend->matmulDev($aId, $bId, $cId, $M, $K, $N);
$backend->reluFwdDev($xId, $yId, $maskId, $n);
$backend->softmaxCeDev($logitsId, $targetsId, $dlogitsId, $lossId, $B, $C);
$backend->sgdUpdateDev($wId, $dwId, $lr, $n);

$result = $backend->bufferDownload($cId, $nFloats);
$backend->bufferFree($wId);
```

### Persistent device tensors (GPU)

Same API, but with `CudaBackend`. Identical method names.

### Audio

```php
$wav = Wav::fromFile('clip.wav');
$mel = new MelSpectrogram($backend, sampleRate: 16000, fftSize: 512, hopSize: 160, nMels: 64);
$tensor = $mel->fromWav($wav);   // [1, 64, nFrames]
```

---

## Conventions (important)

1. **Never overwrite the user's real repo blindly.** The user has a
   mature repo. Ask before replacing large files.

2. **Source lives in git; binaries do not.** `.so` files are built
   per-machine. `.gitignore` covers `native/*/*.so`, `logs/`, `checkpoints/`, `data/`.

3. **Colab + laptop workflow.** Laptop edits → commit → push. Colab
   pulls, builds, tests. Never edit on both simultaneously without
   pulling first.

4. **Numerical correctness first.** Every gradient op is checked
   against finite differences. Don't skip this for new ops.

5. **Reproducible benchmarks.** Every number in the README includes
   hardware, PHP version, and config. Never claim performance without
   measurement.

6. **Honest framing.** ZillaPHP is not "faster than PyTorch." It's
   "competitive with PyTorch on GPU-bound workloads, and practically
   usable on CPU." Don't overclaim.

---

## Session Continuity Notes

**What a new AI session should know:**

- The user runs Linux Mint on an Acer TravelMate (4-core, 8 GB RAM,
  Intel HD 620 — no NVIDIA GPU).
- GPU testing happens on Google Colab (free T4).
- The user is the sole developer and lives in Nigeria.
- The project is academic — there's a paper (`ZillaPHP-Paper.pdf`)
  describing the architecture and hypotheses.
- H0 (framework overhead makes PHP non-competitive) has been **rejected**
  by measurement on both CPU and GPU.
- H1 (native backends make PHP competitive) has been **confirmed**.

**User's working style:**
- Prefers step-by-step instructions with exact commands
- Uses VS Code on laptop + Colab notebooks for GPU
- Wants full file contents, not diffs (easier to paste)
- Copies work into VS Code, then commits + pushes to GitHub

**What to say if continuing a session:**

> "Continuing ZillaPHP. Last session achieved CPU persistent tensors
> (77× speedup, H0 rejected). Current priorities: (1) fix the sampler
> bug in `zilla generate:text`, (2) add CUDA Conv2D/MaxPool kernels,
> (3) update the paper with the new benchmark results."

---

## Sample Prompts For A New Session

**To continue development:**
> I'm working on ZillaPHP at ~/ZillaPHP. Last session we added persistent
> CPU tensors and got a 77× speedup on MNIST (H0 rejected). Next I want
> to fix the sampler bug where `zilla generate:text --tokens=300` only
> emits ~35 characters. Here's the current state: [paste SESSION_HANDOFF.md]

**To work on a specific feature:**
> Continuing ZillaPHP. I want to add CUDA Conv2D and MaxPool kernels so
> I can train CNN models on GPU. Here's the current state: [paste handoff]

**To review / debug:**
> Continuing ZillaPHP. Here's the current state: [paste handoff].
> I'm hitting this error: [paste error].

---

## Appendix — Where To Find Things

| Need | File |
|---|---|
| Build CPU native lib | `native/cpu/build.sh` |
| Build CUDA native lib | `native/cuda/build.sh` |
| GPU MNIST training | `examples/mnist_gpu_v2.php` |
| CPU MNIST (persistent) | `examples/mnist_cpu_v2.php` |
| Shakespeare training | `examples/train_shakespeare.php` |
| Text generation | `bin/zilla generate:text` |
| Audio demo | `examples/audio_mel_test.php` |
| CUDA kernel tests | `tests/cuda_*.php` |
| Paper | `ZillaPHP-Paper.pdf` |
| Developer guide | `docs/DEVELOPER_GUIDE.md` |
| This file | `docs/SESSION_HANDOFF.md` |

---

**End of handoff.**
```

---

## 👉 Your Three Actions

1. **Save the handoff doc** to your repo:
   ```bash
   cd ~/ZillaPHP
   mkdir -p docs
   # In VS Code: create docs/SESSION_HANDOFF.md, paste content above
   git add docs/SESSION_HANDOFF.md README.md
   git commit -m "docs: session handoff guide + CPU persistent results"
   git pull --rebase origin main && git push origin main
   ```

2. **Copy the handoff doc to your own notes** (Google Drive, Notion, wherever). That way even if GitHub is down or you lose the repo, you have it.

3. **When you start a fresh chat**, paste this in the first message:
   > *"Continuing ZillaPHP. Here is the session handoff: [paste SESSION_HANDOFF.md]. Last session achieved CPU persistent tensors (77× speedup, H0 rejected). Let's work on [what you want]."*

---

## 📋 Today's Session Summary

**What we accomplished:**

| Achievement | Measurement |
|-------------|-------------|
| CUDA training ops in C | ✅ 14 new symbols |
| CPU buffer registry | ✅ 5 new symbols |
| CPU training primitives | ✅ 11 new symbols |
| **CPU persistent MNIST** | ✅ **96.86% in 52 s (77× speedup)** |
| **H0 rejected on CPU** | ✅ Proven |
| README updated | ✅ |
| Session handoff doc | ✅ |

**Two days:** 100× GPU speedup + 77× CPU speedup + H0 rejection.

That's a real research contribution. Rest well. 🚀