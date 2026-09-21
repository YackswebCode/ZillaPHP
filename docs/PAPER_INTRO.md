# 1. Introduction

## 1.1 The State of Machine Learning Tooling

Artificial intelligence has evolved from specialized statistical algorithms into
a broad computational discipline that now underpins computer vision, natural
language processing, speech recognition, generative modeling, and increasingly
complex multimodal systems. The frameworks that support this work — PyTorch
[1], TensorFlow [2], and JAX [3] — have become the dominant interfaces through
which researchers and practitioners express, train, and deploy machine learning
models.

These frameworks share a common architectural principle: an expressive
high-level programming interface (typically Python) backed by numerically
intensive native implementations (typically C++ or CUDA). PyTorch demonstrated
that an imperative programming model can combine developer usability with
efficient hardware-accelerated execution. TensorFlow established a
computation-oriented framework capable of targeting heterogeneous
environments. JAX showed that composable transformations — automatic
differentiation, vectorization, and just-in-time compilation — could be
expressed as orthogonal primitives.

Together, these frameworks have substantially reduced the amount of low-level
infrastructure that machine learning practitioners must build. However, their
programming interfaces are **concentrated around a small set of host
languages**: Python, C++, and a handful of JVM-based languages.

## 1.2 The PHP Ecosystem Gap

PHP remains one of the most widely deployed server-side languages in the world.
A substantial fraction of the public web — content management systems, web
applications, APIs, financial systems, enterprise platforms, and e-commerce
infrastructure — is implemented in PHP. Frameworks such as Laravel, Symfony,
and WordPress have built large developer communities and mature tooling
ecosystems around the language.

When these applications require machine learning functionality — sentiment
analysis, recommendation, image classification, speech processing, or any of
the tasks that modern AI enables — developers face a structural discontinuity:
**the machine learning ecosystem and the PHP ecosystem do not share a
programming interface**.

The dominant workarounds are functional but architecturally awkward:

1. **Microservice extraction.** Deploy a Python service (often behind Flask or
   FastAPI) alongside the PHP application. This introduces operational
   complexity: two runtimes, two deployment pipelines, network latency, and
   duplication of authentication and data models.

2. **REST API integration.** Call a hosted inference API (OpenAI, Hugging Face
   Inference, cloud ML platforms). This outsources the model but introduces
   cost, latency, privacy, and dependency concerns.

3. **PHP-to-Python bindings.** Use extensions such as `php-ml` or
   `php-ffi`-based bridges. These are typically narrow, covering only
   classical ML (regression, decision trees) rather than modern tensor-based
   deep learning.

4. **Command-line integration.** Invoke a Python script or a compiled binary
   from PHP via `shell_exec`. This works for batch processing but not for
   latency-sensitive or stateful workloads.

In each case, the PHP developer is **required to leave the PHP ecosystem** in
order to access machine learning. The result is a persistent gap between the
application layer, where domain logic lives, and the machine learning layer,
where numerical computation happens.

## 1.3 Research Question

This paper asks a narrower question than "can PHP replace Python for machine
learning." It asks:

> **Can a PHP-first machine learning framework — one in which the public API
> and the model definition language are PHP, while computationally intensive
> operations are delegated to optimized native backends — provide a practical
> interface for modern tensor-based machine learning?**

The question is empirical, not rhetorical. It requires building a framework
and measuring whether the architecture works: whether autograd can be written
in PHP, whether native delegation delivers meaningful speedups, whether the
framework can train real models on real datasets, and whether it can be
extended to new hardware backends without altering the developer-facing API.

## 1.4 The ZillaPHP Approach

ZillaPHP is a proposed open-source, PHP-native framework for machine learning
and artificial intelligence. It is designed around a layered architecture in
which:

- **PHP provides the developer-facing framework.** Model definitions, training
  loops, dataset handling, and the operator API are all expressed as ordinary
  PHP classes.

- **Native backends provide the numerical execution.** Matrix multiplication,
  element-wise operations, activation functions, convolution, and pooling are
  delegated to compiled C libraries accessed via PHP's Foreign Function
  Interface (FFI). The same architecture permits CUDA and, eventually, ROCm
  backends without changing the PHP API.

- **The backend is selected at runtime.** A `Backend` interface defines the
  operator contract; `CpuBackend`, `NativeCpuBackend`, and `CudaBackend` are
  interchangeable implementations. Model code is agnostic to which backend
  executes it.

The architectural principle is therefore:

    PHP Model  →  Tensor API  →  Runtime  →  Operator Dispatcher  →  Backend
                                                                       ↓
                                                        CPU (C) / CUDA / ROCm

The developer writes PHP. The backend handles numerical execution.

This is not a claim that PHP is a suitable language for writing high-performance
numerical kernels. It is a claim that **the developer-facing language and the
numerical execution language can be separated more sharply than is common**.
PyTorch already does this separation between Python and C++/CUDA. ZillaPHP
extends the same idea to a different developer-facing language.

## 1.5 Contributions

This paper makes the following contributions:

**1. A PHP-first framework architecture.** We present a layered design in
which the tensor abstraction, automatic differentiation engine, neural-network
module system, and training loop are all implemented in PHP, while
computationally intensive operators are dispatched to a pluggable backend
interface.

**2. A verified reverse-mode autograd engine.** Every differentiable
operation in the framework — including non-trivial cases such as
`layerNorm`, `conv2d`, `maxPool2d`, and `softmax` — is verified against
numerical finite differences with tolerances matched to float32 precision.
The full test suite passes on the release cited in this paper.

**3. A native C backend exposed via PHP FFI.** Without requiring a custom PHP
extension or a compilation step for the user, the framework loads a compiled
C shared library at runtime and delegates matrix multiplication, element-wise
operations, activation functions, convolution, and pooling to it. We report
up to **209× speedup** over the pure-PHP implementation on 512×512 matrix
multiplication, with progressively larger gains as operation size increases.

**4. A functional Transformer implementation.** The framework includes
multi-head attention, causal masking, positional embeddings, layer
normalization, and a character-level tokenizer. A small Transformer
language model trained on the Tiny Shakespeare corpus achieves a
cross-entropy loss of approximately **2.40** with 76k parameters and
**~1.75–1.85** with 825k parameters, with working autoregressive generation
and a KV cache delivering a **15× speedup** in token-by-token sampling.

**5. End-to-end training results on public benchmarks.** We report
**98.05%** accuracy on full MNIST (60,000 samples) with a
fully-connected network, and **98.13%** with a small convolutional network,
using the framework's own autograd, optimizer, and native kernels.

**6. A CUDA backend and an empirical finding on GPU offload.** The framework
implements a tiled GEMM kernel, element-wise ops, and activation functions in
CUDA, tested on an NVIDIA Tesla T4. We report an honest result: on
virtualized cloud GPUs, **per-operation GPU offload provides little or no
speedup over native CPU for small-to-medium tensors** because driver and
transfer latency dominate. This is a documented characteristic of GRID-based
virtualization, and it motivates the persistent-device-tensor direction
we discuss in the future-work section.

**7. Reproducibility-first methodology.** All benchmarks, training runs, and
gradient checks are reproducible from the source repository using the
commands cited throughout. We report environment details, tolerance choices,
and limitations explicitly.

## 1.6 What This Paper Does Not Claim

Consistent with the framework's own development philosophy, we are explicit
about what is **not** claimed:

- ZillaPHP is **not** a replacement for PyTorch or TensorFlow. It is
  architecturally smaller and its performance on large models is not
  competitive with established frameworks on the same hardware.

- The framework does **not** claim state-of-the-art accuracy. The 98.05% MNIST
  MLP result is approximately 0.85 percentage points below the reference
  result from LeCun et al. (1998); the 98.13% CNN result uses a very small
  network (~9k parameters) that is capacity-limited rather than
  benchmark-optimal.

- The CUDA backend does **not** claim to demonstrate the achievable GPU
  speedup on bare-metal hardware. Our numbers reflect the behavior of
  virtualized cloud GPUs, which we report as found.

- We do **not** claim that PHP is the appropriate language for writing
  numerical kernels. We claim that it can be the appropriate language for
  expressing models, provided that kernels are delegated elsewhere.

The contribution of the paper is therefore **architectural and empirical**,
not competitive. It asks whether a specific separation of concerns —
PHP for the developer, C/CUDA for the numerics — can produce a usable
machine learning framework in practice. The answer, based on the results
reported here, is a qualified yes.

## 1.7 Paper Structure

The remainder of this paper is organized as follows.

**Part I** introduces the research problem, related work, and the design
philosophy that motivates ZillaPHP.

**Part II** presents the system architecture: the tensor abstraction, the
automatic differentiation engine, the neural-network module system, the
attention and Transformer subsystems, and the multimodal tensor foundation.

**Part III** describes the runtime and performance architecture, including
the operator dispatcher, the backend interface, the native C and CUDA
implementations, and the memory-management considerations that shape the
performance strategy.

**Part IV** documents the developer experience: the command-line interface,
the project structure, and the model serialization format.

**Part V** describes the open-source governance model, contribution workflow,
and the release process.

**Part VI** presents the experimental methodology and reports the results
summarized in §1.5, with full benchmark tables, gradient-check results, and
threats to validity.

**Part VII** outlines the implementation roadmap, including the phases
already completed and the remaining directions toward a stable 1.0 release.

**Part VIII** discusses limitations, expected contributions, and the broader
research question of how programming-language ecosystems influence the design
and accessibility of machine learning tooling.

The appendices provide a quick-reference table of framework components, a
glossary, and an implementation checklist for reproducing the results
reported in this paper.

---

*This is a draft of §1 of the ZillaPHP paper. It will be refined as later
sections are written. Numbers reported here correspond to the results in
`docs/RESULTS.md`, version 1.1.*