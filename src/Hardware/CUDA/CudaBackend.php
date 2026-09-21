<?php

declare(strict_types=1);

namespace ZillaPHP\Hardware\CUDA;

use FFI;
use FFI\CData;
use ZillaPHP\Hardware\CPU\CpuBackend;
use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

/**
 * CUDA backend. Delegates matmul and elementwise operations to a
 * compiled CUDA shared library loaded via PHP FFI.
 *
 * Falls back to the CPU backend transparently when:
 *   - CUDA is not available on this machine
 *   - the shared library is missing
 *   - an operation has no CUDA implementation yet
 *
 * Two execution paths are supported:
 *
 *   1. Naive path — data goes PHP → C buffer → GPU → C buffer → PHP on
 *      every call. Simple but PCIe-bound. Used by matmul(), relu(), etc.
 *
 *   2. Persistent path — data lives on the GPU between calls, referenced
 *      by integer handle. Only upload once at the start and download once
 *      at the end. See bufferAlloc/bufferUpload/matmulDev.
 *
 * IMPORTANT: FFI + CUDA interop requires that the CUDA library exports
 * plain C symbols (extern "C"). All kernel launchers do this.
 */
final class CudaBackend extends CpuBackend
{
    private ?FFI $ffi = null;
    private bool $loaded = false;

    public function __construct(private ?string $libraryPath = null)
    {
        $this->libraryPath ??= __DIR__ . '/../../../native/cuda/libzilla_cuda.so';

        if (!extension_loaded('FFI')) {
            return;
        }
        if (!is_file($this->libraryPath)) {
            return;
        }

        try {
            $this->ffi = FFI::cdef(
                <<<'C'
                void matmul_f32(const float* A, const float* B, float* C, int M, int K, int N);
                void relu_f32_cuda(const float* X, float* Y, int N);
                void add_f32_cuda(const float* A, const float* B, float* C, int N);
                void sub_f32_cuda(const float* A, const float* B, float* C, int N);
                void mul_f32_cuda(const float* A, const float* B, float* C, int N);
                void div_f32_cuda(const float* A, const float* B, float* C, int N);
                void hann_window_cuda(float* w, int N);
                void stft_magnitude_cuda(const float* input, int inputLen, const float* window, int fftSize, int hopSize, float* output, int* outNBins, int* outNFrames);
                int  zilla_buffer_alloc(unsigned long n_floats);
                void zilla_buffer_free(int id);
                int  zilla_buffer_upload(int id, const float* host, unsigned long n_floats);
                int  zilla_buffer_download(int id, float* host, unsigned long n_floats);
                void zilla_sync(void);
                int  matmul_dev(int a_id, int b_id, int c_id, int M, int K, int N);
                int  add_dev(int a_id, int b_id, int c_id, int n);
                int  sub_dev(int a_id, int b_id, int c_id, int n);
                int  mul_dev(int a_id, int b_id, int c_id, int n);
                int  div_dev(int a_id, int b_id, int c_id, int n);
                int  relu_dev(int x_id, int y_id, int n);
                int  add_bias_dev(int x_id, int b_id, int B, int C);
                int  relu_fwd_dev(int x_id, int y_id, int mask_id, int n);
                int  relu_bwd_dev(int dy_id, int mask_id, int dx_id, int n);
                int  softmax_ce_dev(int logits_id, int targets_id, int dlogits_id, int loss_id, int B, int C);
                int  bias_grad_dev(int dy_id, int db_id, int B, int C);
                int  sgd_update_dev(int w_id, int dw_id, float lr, int n);
                int  zero_dev(int id, int n);
                int  matmul_tn_dev(int a_id, int b_id, int c_id, int M, int K, int N);
                int  matmul_nt_dev(int a_id, int b_id, int c_id, int M, int K, int N);
                int  copy_dev(int src_id, int dst_id, int n);
                C,
                $this->libraryPath
            );
            $this->loaded = true;
        } catch (\Throwable) {
            $this->loaded = false;
        }
    }

    public function name(): string
    {
        return $this->loaded ? 'cuda' : 'cpu';
    }

    public function isCuda(): bool
    {
        return $this->loaded;
    }

    // ==================================================================
    // Buffer helpers
    // ==================================================================

    /**
     * Copy a PHP array of numbers into a freshly allocated C float array.
     *
     * Writes element-by-element through FFI's accessor rather than using
     * pack('f*', ...$arr) — the spread operator is O(n²) in PHP for large
     * arrays and dominates runtime on multi-hundred-KB tensors.
     *
     * @param array<int|float|bool> $arr
     */
    private function toC(array $arr): CData
    {
        $n   = count($arr);
        $buf = $this->ffi->new("float[{$n}]");
        for ($i = 0; $i < $n; $i++) {
            $buf[$i] = (float) $arr[$i];
        }
        return $buf;
    }

    /**
     * Read a C float array back into a PHP array.
     *
     * @return float[]
     */
    private function fromC(CData $buf, int $n): array
    {
        $arr = [];
        for ($i = 0; $i < $n; $i++) {
            $arr[] = $buf[$i];
        }
        return $arr;
    }

    // ==================================================================
    // Naive-path kernels (transfer on every call)
    // ==================================================================

    public function matmul(Tensor $a, Tensor $b): Tensor
    {
        if (!$this->loaded) return parent::matmul($a, $b);

        $ashape = $a->shape()->dims();
        $bshape = $b->shape()->dims();

        if (count($ashape) !== 2 || count($bshape) !== 2) {
            return parent::matmul($a, $b);
        }

        [$m, $k]  = $ashape;
        [$k2, $n] = $bshape;

        if ($k !== $k2) {
            throw new \InvalidArgumentException(
                "matmul shape mismatch: ({$m}x{$k}) @ ({$k2}x{$n})"
            );
        }

        // Very small matrices: not worth the GPU round trip
        if ($m * $k * $n < 100000) {
            return parent::matmul($a, $b);
        }

        $A = $this->toC($a->data());
        $B = $this->toC($b->data());
        $C = $this->ffi->new("float[" . ($m * $n) . "]");

        $this->ffi->matmul_f32($A, $B, $C, $m, $k, $n);

        return new Tensor(
            $this->fromC($C, $m * $n),
            new Shape([$m, $n]),
            $a->dtype(),
            $a->device()
        );
    }

    public function relu(Tensor $tensor): Tensor
    {
        if (!$this->loaded) return parent::relu($tensor);

        $n = $tensor->shape()->size();
        if ($n < 4096) return parent::relu($tensor);

        $X = $this->toC($tensor->data());
        $Y = $this->ffi->new("float[{$n}]");

        $this->ffi->relu_f32_cuda($X, $Y, $n);

        return new Tensor(
            $this->fromC($Y, $n),
            $tensor->shape(),
            $tensor->dtype(),
            $tensor->device()
        );
    }

    public function add(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwiseCuda($a, $b, 'add_f32_cuda', 'add');
    }

    public function subtract(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwiseCuda($a, $b, 'sub_f32_cuda', 'subtract');
    }

    public function multiply(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwiseCuda($a, $b, 'mul_f32_cuda', 'multiply');
    }

    public function divide(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwiseCuda($a, $b, 'div_f32_cuda', 'divide');
    }

    private function elementwiseCuda(
        Tensor $a,
        Tensor $b,
        string $cFunc,
        string $parentMethod,
    ): Tensor {
        if (!$this->loaded) {
            /** @var Tensor $r */
            $r = parent::$parentMethod($a, $b);
            return $r;
        }
        if ($a->shape()->dims() !== $b->shape()->dims()) {
            /** @var Tensor $r */
            $r = parent::$parentMethod($a, $b);
            return $r;
        }

        $n = $a->shape()->size();
        if ($n < 4096) {
            /** @var Tensor $r */
            $r = parent::$parentMethod($a, $b);
            return $r;
        }

        $A = $this->toC($a->data());
        $B = $this->toC($b->data());
        $C = $this->ffi->new("float[{$n}]");

        match ($cFunc) {
            'add_f32_cuda' => $this->ffi->add_f32_cuda($A, $B, $C, $n),
            'sub_f32_cuda' => $this->ffi->sub_f32_cuda($A, $B, $C, $n),
            'mul_f32_cuda' => $this->ffi->mul_f32_cuda($A, $B, $C, $n),
            'div_f32_cuda' => $this->ffi->div_f32_cuda($A, $B, $C, $n),
        };

        return new Tensor(
            $this->fromC($C, $n),
            $a->shape(),
            $a->dtype(),
            $a->device()
        );
    }

    // ==================================================================
    // Audio: Hann window + STFT (cuFFT-backed)
    // ==================================================================

    /**
     * Generate a Hann window of length N on the GPU.
     *
     * @return float[]  length-N array, w[0] = w[N-1] = 0, w[N/2] = 1
     */
    public function hannWindow(int $n): array
    {
        if (!$this->loaded) {
            throw new \RuntimeException("CUDA backend not loaded.");
        }
        if ($n < 1) {
            throw new \InvalidArgumentException("Window length must be >= 1.");
        }

        $w = $this->ffi->new("float[{$n}]");
        $this->ffi->hann_window_cuda($w, $n);
        return $this->fromC($w, $n);
    }

    /**
     * STFT magnitude of a mono waveform using cuFFT.
     *
     * Note: the $window argument is currently ignored — the CUDA kernel
     * regenerates the Hann window on-device and caches it. Kept in the
     * signature for API compatibility with NativeCpuBackend.
     *
     * @param float[] $input    audio samples in [-1, 1]
     * @param float[] $window   Hann window (see note above)
     * @param int     $fftSize  power of two, <= 16384
     * @param int     $hopSize  frame hop
     *
     * @return array{0: float[], 1: int, 2: int}
     *         [magnitude, nBins, nFrames]
     *         magnitude is row-major [nBins, nFrames],
     *         access via mag[bin * nFrames + frame]
     */
    public function stftMagnitude(
        array $input,
        array $window,
        int $fftSize,
        int $hopSize,
    ): array {
        if (!$this->loaded) {
            throw new \RuntimeException("CUDA backend not loaded.");
        }
        if (count($window) !== $fftSize) {
            throw new \InvalidArgumentException(
                "Window length (" . count($window) . ") must equal fftSize ({$fftSize})."
            );
        }
        if ($fftSize < 2 || ($fftSize & ($fftSize - 1)) !== 0) {
            throw new \InvalidArgumentException("fftSize must be a power of two.");
        }
        if ($fftSize > 16384) {
            throw new \InvalidArgumentException("fftSize must be <= 16384.");
        }
        if ($hopSize < 1) {
            throw new \InvalidArgumentException("hopSize must be >= 1.");
        }

        $inputLen = count($input);
        $inC      = $this->toC($input);
        $winC     = $this->toC($window);   // passed for signature parity; kernel ignores

        // Upper bound on (nBins * nFrames)
        $nBins     = intdiv($fftSize, 2) + 1;
        $maxFrames = $inputLen < $fftSize
            ? 1
            : 1 + intdiv($inputLen - $fftSize, $hopSize);
        $outSize   = $nBins * max($maxFrames, 1);

        $outC   = $this->ffi->new("float[{$outSize}]");
        $nBinsC = $this->ffi->new("int[1]");
        $nFrC   = $this->ffi->new("int[1]");

        $this->ffi->stft_magnitude_cuda(
            $inC, $inputLen, $winC, $fftSize, $hopSize,
            $outC, $nBinsC, $nFrC,
        );

        $nBinsOut   = (int) $nBinsC[0];
        $nFramesOut = (int) $nFrC[0];

        if ($nBinsOut === 0 || $nFramesOut === 0) {
            throw new \RuntimeException(
                "CUDA STFT failed — check fftSize, window length, and hopSize."
            );
        }

        return [
            $this->fromC($outC, $nBinsOut * $nFramesOut),
            $nBinsOut,
            $nFramesOut,
        ];
    }

    // ==================================================================
    // Persistent device buffers (Option C)
    // ==================================================================
    //
    // Workflow for GPU-resident computation:
    //
    //   $aId = $backend->bufferAlloc($n);
    //   $bId = $backend->bufferAlloc($n);
    //   $cId = $backend->bufferAlloc($n);
    //   $backend->bufferUpload($aId, $a);
    //   $backend->bufferUpload($bId, $b);
    //   $backend->matmulDev($aId, $bId, $cId, $n, $n, $n);
    //   $result = $backend->bufferDownload($cId, $n);
    //   $backend->bufferFree($aId);
    //   $backend->bufferFree($bId);
    //   $backend->bufferFree($cId);
    //
    // For multi-op pipelines, keep the handles alive and run many *_dev
    // calls before downloading. This is where the 15-50x speedup lives.

    /**
     * Allocate a device buffer holding $nFloats floats.
     *
     * @return int  Positive buffer handle, or throws on failure.
     */
    public function bufferAlloc(int $nFloats): int
    {
        if (!$this->loaded) {
            throw new \RuntimeException("CUDA backend not loaded.");
        }
        if ($nFloats < 1) {
            throw new \InvalidArgumentException("nFloats must be >= 1.");
        }

        $id = $this->ffi->zilla_buffer_alloc($nFloats);
        if ($id < 0) {
            throw new \RuntimeException(
                "zilla_buffer_alloc({$nFloats}) failed — GPU out of memory?"
            );
        }
        return (int) $id;
    }

    /**
     * Free a device buffer. Idempotent: freeing an already-freed or
     * invalid handle is a safe no-op.
     */
    public function bufferFree(int $id): void
    {
        if (!$this->loaded) return;
        if ($id > 0) {
            $this->ffi->zilla_buffer_free($id);
        }
    }

    /**
     * Copy a PHP float array into a previously allocated device buffer.
     *
     * @param float[] $host
     */
    public function bufferUpload(int $id, array $host): void
    {
        if (!$this->loaded) {
            throw new \RuntimeException("CUDA backend not loaded.");
        }
        if ($id < 1) {
            throw new \InvalidArgumentException("Invalid buffer id: {$id}");
        }

        $buf = $this->toC($host);
        $ok  = $this->ffi->zilla_buffer_upload($id, $buf, count($host));
        if ($ok !== 0) {
            throw new \RuntimeException(
                "zilla_buffer_upload failed for id {$id} (" . count($host) . " floats)."
            );
        }
    }

    /**
     * Copy a device buffer back into a PHP float array.
     *
     * @return float[]
     */
    public function bufferDownload(int $id, int $nFloats): array
    {
        if (!$this->loaded) {
            throw new \RuntimeException("CUDA backend not loaded.");
        }
        if ($id < 1) {
            throw new \InvalidArgumentException("Invalid buffer id: {$id}");
        }
        if ($nFloats < 1) {
            throw new \InvalidArgumentException("nFloats must be >= 1.");
        }

        $buf = $this->ffi->new("float[{$nFloats}]");
        $ok  = $this->ffi->zilla_buffer_download($id, $buf, $nFloats);
        if ($ok !== 0) {
            throw new \RuntimeException("zilla_buffer_download failed for id {$id}.");
        }
        return $this->fromC($buf, $nFloats);
    }

    /**
     * Block until all pending GPU work is complete. Useful for timing.
     */
    public function sync(): void
    {
        if (!$this->loaded) return;
        $this->ffi->zilla_sync();
    }

    /**
     * Device-to-device matmul:
     *   C = A @ B
     * All three handles must be pre-allocated. Sizes in floats.
     */
    public function matmulDev(int $aId, int $bId, int $cId, int $m, int $k, int $n): void
    {
        if (!$this->loaded) {
            throw new \RuntimeException("CUDA backend not loaded.");
        }
        $ok = $this->ffi->matmul_dev($aId, $bId, $cId, $m, $k, $n);
        if ($ok !== 0) {
            throw new \RuntimeException(
                "matmul_dev({$aId}, {$bId}, {$cId}, {$m}, {$k}, {$n}) failed."
            );
        }
    }

    /**
     * Device-to-device elementwise ops.
     */
    public function addDev(int $aId, int $bId, int $cId, int $n): void
    {
        if (!$this->loaded) throw new \RuntimeException("CUDA backend not loaded.");
        if ($this->ffi->add_dev($aId, $bId, $cId, $n) !== 0) {
            throw new \RuntimeException("add_dev failed.");
        }
    }

    public function subDev(int $aId, int $bId, int $cId, int $n): void
    {
        if (!$this->loaded) throw new \RuntimeException("CUDA backend not loaded.");
        if ($this->ffi->sub_dev($aId, $bId, $cId, $n) !== 0) {
            throw new \RuntimeException("sub_dev failed.");
        }
    }

    public function mulDev(int $aId, int $bId, int $cId, int $n): void
    {
        if (!$this->loaded) throw new \RuntimeException("CUDA backend not loaded.");
        if ($this->ffi->mul_dev($aId, $bId, $cId, $n) !== 0) {
            throw new \RuntimeException("mul_dev failed.");
        }
    }

    public function divDev(int $aId, int $bId, int $cId, int $n): void
    {
        if (!$this->loaded) throw new \RuntimeException("CUDA backend not loaded.");
        if ($this->ffi->div_dev($aId, $bId, $cId, $n) !== 0) {
            throw new \RuntimeException("div_dev failed.");
        }
    }

    public function reluDev(int $xId, int $yId, int $n): void
    {
        if (!$this->loaded) throw new \RuntimeException("CUDA backend not loaded.");
        if ($this->ffi->relu_dev($xId, $yId, $n) !== 0) {
            throw new \RuntimeException("relu_dev failed.");
        }
    }

        // ==================================================================
    // Training primitives (v2)
    // ==================================================================

    public function addBiasDev(int $xId, int $bId, int $B, int $C): void
    {
        if ($this->ffi->add_bias_dev($xId, $bId, $B, $C) !== 0)
            throw new \RuntimeException("add_bias_dev failed.");
    }

    public function reluFwdDev(int $xId, int $yId, int $maskId, int $n): void
    {
        if ($this->ffi->relu_fwd_dev($xId, $yId, $maskId, $n) !== 0)
            throw new \RuntimeException("relu_fwd_dev failed.");
    }

    public function reluBwdDev(int $dyId, int $maskId, int $dxId, int $n): void
    {
        if ($this->ffi->relu_bwd_dev($dyId, $maskId, $dxId, $n) !== 0)
            throw new \RuntimeException("relu_bwd_dev failed.");
    }

    public function softmaxCeDev(int $logitsId, int $targetsId, int $dlogitsId, int $lossId, int $B, int $C): void
    {
        if ($this->ffi->softmax_ce_dev($logitsId, $targetsId, $dlogitsId, $lossId, $B, $C) !== 0)
            throw new \RuntimeException("softmax_ce_dev failed.");
    }

    public function biasGradDev(int $dyId, int $dbId, int $B, int $C): void
    {
        if ($this->ffi->bias_grad_dev($dyId, $dbId, $B, $C) !== 0)
            throw new \RuntimeException("bias_grad_dev failed.");
    }

    public function sgdUpdateDev(int $wId, int $dwId, float $lr, int $n): void
    {
        if ($this->ffi->sgd_update_dev($wId, $dwId, $lr, $n) !== 0)
            throw new \RuntimeException("sgd_update_dev failed.");
    }

    public function zeroDev(int $id, int $n): void
    {
        if ($this->ffi->zero_dev($id, $n) !== 0)
            throw new \RuntimeException("zero_dev failed.");
    }

    public function matmulTnDev(int $aId, int $bId, int $cId, int $M, int $K, int $N): void
    {
        if ($this->ffi->matmul_tn_dev($aId, $bId, $cId, $M, $K, $N) !== 0)
            throw new \RuntimeException("matmul_tn_dev failed.");
    }

    public function matmulNtDev(int $aId, int $bId, int $cId, int $M, int $K, int $N): void
    {
        if ($this->ffi->matmul_nt_dev($aId, $bId, $cId, $M, $K, $N) !== 0)
            throw new \RuntimeException("matmul_nt_dev failed.");
    }

    public function copyDev(int $srcId, int $dstId, int $n): void
    {
        if ($this->ffi->copy_dev($srcId, $dstId, $n) !== 0)
            throw new \RuntimeException("copy_dev failed.");
    }
}