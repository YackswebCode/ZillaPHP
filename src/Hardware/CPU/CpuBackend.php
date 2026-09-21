<?php

declare(strict_types=1);

namespace ZillaPHP\Hardware\CPU;

use ZillaPHP\Core\Contracts\Backend;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Tensor\Shape\Shape;

class CpuBackend implements Backend
{
    public function name(): string
    {
        return 'cpu';
    }

    private function assertSameShape(Tensor $a, Tensor $b): void
    {
        if ($a->shape()->dims() !== $b->shape()->dims()) {
            throw new \InvalidArgumentException(
                "Shape mismatch: {$a->shape()} vs {$b->shape()}"
            );
        }
    }

    private function elementwise(Tensor $a, Tensor $b, callable $op): Tensor
    {
        $this->assertSameShape($a, $b);
        $out = [];
        $ad = $a->data();
        $bd = $b->data();
        $n = count($ad);
        for ($i = 0; $i < $n; $i++) {
            $out[] = $op($ad[$i], $bd[$i]);
        }
        return new Tensor($out, $a->shape(), $a->dtype(), $a->device());
    }

    public function add(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwise($a, $b, fn($x, $y) => $x + $y);
    }

    public function subtract(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwise($a, $b, fn($x, $y) => $x - $y);
    }

    public function multiply(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwise($a, $b, fn($x, $y) => $x * $y);
    }

    public function divide(Tensor $a, Tensor $b): Tensor
    {
        return $this->elementwise($a, $b, fn($x, $y) => $x / $y);
    }

    public function matmul(Tensor $a, Tensor $b): Tensor
    {
        $ashape = $a->shape()->dims();
        $bshape = $b->shape()->dims();

        if (count($ashape) !== 2 || count($bshape) !== 2) {
            throw new \InvalidArgumentException("matmul currently supports 2D tensors only.");
        }

        [$m, $k]  = $ashape;
        [$k2, $n] = $bshape;

        if ($k !== $k2) {
            throw new \InvalidArgumentException(
                "matmul shape mismatch: ({$m}x{$k}) @ ({$k2}x{$n})"
            );
        }

        $ad = $a->data();
        $bd = $b->data();
        $out = array_fill(0, $m * $n, 0.0);

        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $sum = 0.0;
                for ($p = 0; $p < $k; $p++) {
                    $sum += $ad[$i * $k + $p] * $bd[$p * $n + $j];
                }
                $out[$i * $n + $j] = $sum;
            }
        }

        return new Tensor($out, new Shape([$m, $n]), $a->dtype(), $a->device());
    }

    public function sum(Tensor $tensor): Tensor
    {
        $total = array_sum($tensor->data());
        return new Tensor([$total], new Shape([1]));
    }

    public function mean(Tensor $tensor): Tensor
    {
        $d = $tensor->data();
        $mean = array_sum($d) / max(count($d), 1);
        return new Tensor([$mean], new Shape([1]));
    }

    public function relu(Tensor $tensor): Tensor
    {
        $out = array_map(fn($x) => max(0.0, $x), $tensor->data());
        return new Tensor($out, $tensor->shape(), $tensor->dtype(), $tensor->device());
    }

    public function exp(Tensor $tensor): Tensor
    {
        $out = array_map(fn($x) => exp($x), $tensor->data());
        return new Tensor($out, $tensor->shape(), $tensor->dtype(), $tensor->device());
    }

    public function log(Tensor $tensor): Tensor
    {
        $out = array_map(fn($x) => log($x), $tensor->data());
        return new Tensor($out, $tensor->shape(), $tensor->dtype(), $tensor->device());
    }
}