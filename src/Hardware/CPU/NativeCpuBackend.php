<?php

declare(strict_types=1);

namespace ZillaPHP\Hardware\CPU;

use FFI;
use FFI\CData;
use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

/**
 * CPU backend that delegates heavy numerical kernels to a compiled
 * C shared library loaded via PHP FFI.
 */
final class NativeCpuBackend extends CpuBackend
{
    private ?FFI $ffi = null;
    private bool $loaded = false;

    public function __construct(private ?string $libraryPath = null)
    {
        $this->libraryPath ??= __DIR__ . '/../../../native/cpu/libzilla_cpu.so';

        if (!extension_loaded('FFI')) {
            return;
        }
        if (!is_file($this->libraryPath)) {
            return;
        }

        try {
            $this->ffi = FFI::cdef(
                <<<'C'
                void  matmul_f32(const float* A, const float* B, float* C, int M, int K, int N);
                void  relu_f32(const float* X, float* Y, int N);
                void  add_f32(const float* A, const float* B, float* C, int N);
                void  sub_f32(const float* A, const float* B, float* C, int N);
                void  mul_f32(const float* A, const float* B, float* C, int N);
                void  div_f32(const float* A, const float* B, float* C, int N);
                float sum_f32(const float* X, int N);
                void  conv2d_forward_f32(const float* input, const float* weight, const float* bias, float* output, int C, int H, int W, int outC, int kH, int kW, int stride, int padding);
                void  conv2d_backward_f32(const float* input, const float* weight, const float* gradOutput, float* gradInput, float* gradWeight, float* gradBias, int C, int H, int W, int outC, int kH, int kW, int stride, int padding);
                void  maxpool2d_forward_f32(const float* input, float* output, int* argmax, int C, int H, int W, int kH, int kW, int stride);
                void  maxpool2d_backward_f32(const float* gradOutput, const int* argmax, float* gradInput, int C, int H, int W, int outH, int outW);
                void  hann_window_f32(float* w, int N);
                void  fft_radix2_f32(float* re, float* im, int N);
                void  stft_magnitude_f32(const float* input, int inputLen, const float* window, int fftSize, int hopSize, float* output, int* outNBins, int* outNFrames);
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
        return $this->loaded ? 'cpu+native' : 'cpu';
    }

    public function isNative(): bool
    {
        return $this->loaded;
    }

    // ==================================================================
    // Buffer conversion helpers
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
    // Basic kernels
    // ==================================================================

    public function matmul(Tensor $a, Tensor $b): Tensor
    {
        if (!$this->loaded) {
            return parent::matmul($a, $b);
        }

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
        if (!$this->loaded) {
            return parent::relu($tensor);
        }

        $n = $tensor->shape()->size();
        $X = $this->toC($tensor->data());
        $Y = $this->ffi->new("float[{$n}]");

        $this->ffi->relu_f32($X, $Y, $n);

        return new Tensor(
            $this->fromC($Y, $n),
            $tensor->shape(),
            $tensor->dtype(),
            $tensor->device()
        );
    }

    public function add(Tensor $a, Tensor $b): Tensor
    {
        if (!$this->loaded) return parent::add($a, $b);
        if ($a->shape()->dims() !== $b->shape()->dims()) return parent::add($a, $b);

        return $this->runElementwise($a, $b, 'add_f32');
    }

    public function subtract(Tensor $a, Tensor $b): Tensor
    {
        if (!$this->loaded) return parent::subtract($a, $b);
        if ($a->shape()->dims() !== $b->shape()->dims()) return parent::subtract($a, $b);

        return $this->runElementwise($a, $b, 'sub_f32');
    }

    public function multiply(Tensor $a, Tensor $b): Tensor
    {
        if (!$this->loaded) return parent::multiply($a, $b);
        if ($a->shape()->dims() !== $b->shape()->dims()) return parent::multiply($a, $b);

        return $this->runElementwise($a, $b, 'mul_f32');
    }

    public function divide(Tensor $a, Tensor $b): Tensor
    {
        if (!$this->loaded) return parent::divide($a, $b);
        if ($a->shape()->dims() !== $b->shape()->dims()) return parent::divide($a, $b);

        return $this->runElementwise($a, $b, 'div_f32');
    }

    public function sum(Tensor $tensor): Tensor
    {
        if (!$this->loaded) {
            return parent::sum($tensor);
        }

        $n = $tensor->shape()->size();
        $X = $this->toC($tensor->data());
        $s = $this->ffi->sum_f32($X, $n);

        return new Tensor([(float) $s], new Shape([1]));
    }

    private function runElementwise(Tensor $a, Tensor $b, string $cFunc): Tensor
    {
        $n = $a->shape()->size();
        $A = $this->toC($a->data());
        $B = $this->toC($b->data());
        $C = $this->ffi->new("float[{$n}]");

        match ($cFunc) {
            'add_f32' => $this->ffi->add_f32($A, $B, $C, $n),
            'sub_f32' => $this->ffi->sub_f32($A, $B, $C, $n),
            'mul_f32' => $this->ffi->mul_f32($A, $B, $C, $n),
            'div_f32' => $this->ffi->div_f32($A, $B, $C, $n),
        };

        return new Tensor(
            $this->fromC($C, $n),
            $a->shape(),
            $a->dtype(),
            $a->device()
        );
    }

    // ==================================================================
    // Conv2D / MaxPool2D
    // ==================================================================

    /**
     * @return float[]
     */
    public function conv2dForward(
        array $inputData,
        array $weightData,
        array $biasData,
        int $C, int $H, int $W,
        int $outC, int $kH, int $kW,
        int $stride, int $padding,
    ): array {
        if (!$this->loaded) {
            throw new \RuntimeException("Native backend not loaded.");
        }

        $outH = intdiv($H + 2 * $padding - $kH, $stride) + 1;
        $outW = intdiv($W + 2 * $padding - $kW, $stride) + 1;

        $inC   = $this->toC($inputData);
        $wC    = $this->toC($weightData);
        $bC    = $this->toC($biasData);
        $outC_ = $this->ffi->new("float[" . ($outC * $outH * $outW) . "]");

        $this->ffi->conv2d_forward_f32(
            $inC, $wC, $bC, $outC_,
            $C, $H, $W,
            $outC, $kH, $kW,
            $stride, $padding,
        );

        return $this->fromC($outC_, $outC * $outH * $outW);
    }

    /**
     * @return array{0: float[], 1: float[], 2: float[]}
     */
    public function conv2dBackward(
        array $inputData,
        array $weightData,
        array $gradOutputData,
        int $C, int $H, int $W,
        int $outC, int $kH, int $kW,
        int $stride, int $padding,
    ): array {
        if (!$this->loaded) {
            throw new \RuntimeException("Native backend not loaded.");
        }

        $inC  = $this->toC($inputData);
        $wC   = $this->toC($weightData);
        $gC   = $this->toC($gradOutputData);
        $giC  = $this->ffi->new("float[" . ($C * $H * $W) . "]");
        $gwC  = $this->ffi->new("float[" . ($outC * $C * $kH * $kW) . "]");
        $gbC  = $this->ffi->new("float[" . $outC . "]");

        $this->ffi->conv2d_backward_f32(
            $inC, $wC, $gC,
            $giC, $gwC, $gbC,
            $C, $H, $W,
            $outC, $kH, $kW,
            $stride, $padding,
        );

        return [
            $this->fromC($giC, $C * $H * $W),
            $this->fromC($gwC, $outC * $C * $kH * $kW),
            $this->fromC($gbC, $outC),
        ];
    }

    /**
     * @return array{0: float[], 1: int[]}
     */
    public function maxPool2dForward(
        array $inputData,
        int $C, int $H, int $W,
        int $kH, int $kW, int $stride,
    ): array {
        if (!$this->loaded) {
            throw new \RuntimeException("Native backend not loaded.");
        }

        $outH = intdiv($H - $kH, $stride) + 1;
        $outW = intdiv($W - $kW, $stride) + 1;

        $inC    = $this->toC($inputData);
        $outC_  = $this->ffi->new("float[" . ($C * $outH * $outW) . "]");
        $argmax = $this->ffi->new("int[" . ($C * $outH * $outW) . "]");

        $this->ffi->maxpool2d_forward_f32(
            $inC, $outC_, $argmax,
            $C, $H, $W,
            $kH, $kW, $stride,
        );

        $outData = $this->fromC($outC_, $C * $outH * $outW);
        $n = $C * $outH * $outW;
        $argArr = [];
        for ($i = 0; $i < $n; $i++) {
            $argArr[] = (int) $argmax[$i];
        }

        return [$outData, $argArr];
    }

    /**
     * @param int[] $argmax
     * @return float[]
     */
    public function maxPool2dBackward(
        array $gradOutputData,
        array $argmax,
        int $C, int $H, int $W,
        int $outH, int $outW,
    ): array {
        if (!$this->loaded) {
            throw new \RuntimeException("Native backend not loaded.");
        }

        $gC = $this->toC($gradOutputData);

        $argC = $this->ffi->new("int[" . count($argmax) . "]");
        foreach ($argmax as $i => $v) {
            $argC[$i] = (int) $v;
        }

        $giC = $this->ffi->new("float[" . ($C * $H * $W) . "]");

        $this->ffi->maxpool2d_backward_f32(
            $gC, $argC, $giC,
            $C, $H, $W,
            $outH, $outW,
        );

        return $this->fromC($giC, $C * $H * $W);
    }

    // ==================================================================
    // Audio: Hann window + STFT
    // ==================================================================

    /**
     * Generate a Hann window of length N (native kernel).
     *
     * @return float[]  length-N array, w[0] = w[N-1] = 0, w[N/2] = 1
     */
    public function hannWindow(int $n): array
    {
        if (!$this->loaded) {
            throw new \RuntimeException("Native backend not loaded.");
        }
        if ($n < 1) {
            throw new \InvalidArgumentException("Window length must be >= 1.");
        }

        $w = $this->ffi->new("float[{$n}]");
        $this->ffi->hann_window_f32($w, $n);
        return $this->fromC($w, $n);
    }

    /**
     * STFT magnitude of a mono waveform.
     *
     * @param float[] $input    audio samples in [-1, 1]
     * @param float[] $window   Hann window of length fftSize
     * @param int     $fftSize  power of two, <= 16384
     * @param int     $hopSize  frame hop
     *
     * @return array{0: float[], 1: int, 2: int}
     *         [magnitude, nBins, nFrames]
     *         magnitude is row-major [nBins, nFrames],
     *         nBins = fftSize/2 + 1, access via mag[bin * nFrames + frame]
     */
    public function stftMagnitude(
        array $input,
        array $window,
        int $fftSize,
        int $hopSize,
    ): array {
        if (!$this->loaded) {
            throw new \RuntimeException("Native backend not loaded.");
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
        $winC     = $this->toC($window);

        // Upper bound on (nBins * nFrames)
        $nBins     = intdiv($fftSize, 2) + 1;
        $maxFrames = $inputLen < $fftSize
            ? 1
            : 1 + intdiv($inputLen - $fftSize, $hopSize);
        $outSize   = $nBins * max($maxFrames, 1);

        $outC   = $this->ffi->new("float[{$outSize}]");
        $nBinsC = $this->ffi->new("int[1]");
        $nFrC   = $this->ffi->new("int[1]");

        $this->ffi->stft_magnitude_f32(
            $inC, $inputLen, $winC, $fftSize, $hopSize,
            $outC, $nBinsC, $nFrC,
        );

        $nBinsOut   = (int) $nBinsC[0];
        $nFramesOut = (int) $nFrC[0];

        if ($nBinsOut === 0 || $nFramesOut === 0) {
            throw new \RuntimeException(
                "STFT failed — check fftSize, window length, and hopSize."
            );
        }

        return [
            $this->fromC($outC, $nBinsOut * $nFramesOut),
            $nBinsOut,
            $nFramesOut,
        ];
    }
}