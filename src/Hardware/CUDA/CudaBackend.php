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

    /** @param array<int|float|bool> $arr */
    private function toC(array $arr): CData
    {
        $n      = count($arr);
        $buf    = $this->ffi->new("float[{$n}]");
        $floats = array_map('floatval', $arr);
        $packed = pack('f*', ...$floats);
        FFI::memcpy($buf, $packed, $n * 4);
        return $buf;
    }

    /** @return float[] */
    private function fromC(CData $buf, int $n): array
    {
        $binary = FFI::string($buf, $n * 4);
        /** @var array<int,float> $unpacked */
        $unpacked = unpack('f*', $binary);
        return array_values($unpacked);
    }

    // ==================================================================
    // Kernels
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
}