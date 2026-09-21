<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Module;
use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Shape\Shape;  
use ZillaPHP\Tensor\Tensor;

final class Conv2D extends Module
{
    private Parameter $weight;  // [outC, inC, kH, kW]
    private ?Parameter $bias;   // [outC] or null

    public function __construct(
        private int $inChannels,
        private int $outChannels,
        private int $kernelSize = 3,
        private int $stride = 1,
        private int $padding = 0,
        bool $useBias = true,
    ) {
        // He initialization: std = sqrt(2 / (inC * kH * kW))
        $fanIn = $inChannels * $kernelSize * $kernelSize;
        $std = sqrt(2.0 / max($fanIn, 1));

        $n = $outChannels * $inChannels * $kernelSize * $kernelSize;
        $data = [];
        for ($i = 0; $i < $n; $i++) {
            $u1 = mt_rand() / mt_getrandmax();
            $u2 = mt_rand() / mt_getrandmax();
            $data[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $std;
        }

        $this->weight = new Parameter(
            $data,
            new Shape([$outChannels, $inChannels, $kernelSize, $kernelSize])
        );

        $this->bias = $useBias
            ? new Parameter(array_fill(0, $outChannels, 0.0), new Shape([$outChannels]))
            : null;
    }

    public function forward(Tensor $input): Tensor
    {
        $out = $input->conv2d($this->weight, $this->bias, $this->stride, $this->padding);
        return $out;
    }

    public function parameters(): array
    {
        return $this->bias === null
            ? [$this->weight]
            : [$this->weight, $this->bias];
    }

    public function weight(): Parameter { return $this->weight; }
    public function bias(): ?Parameter  { return $this->bias; }
}