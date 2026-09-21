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
 *
 * Falls back transparently to pure-PHP CpuBackend if:
 *   - the FFI extension is not enabled
 *   - the shared library is missing
 *   - the current operation is not implemented natively
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
    // Overridden kernels
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
}