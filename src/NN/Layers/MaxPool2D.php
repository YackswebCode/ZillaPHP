<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

final class MaxPool2D extends Module
{
    public function __construct(
        private int $kernelSize = 2,
        private ?int $stride = null,
    ) {}

    public function forward(Tensor $input): Tensor
    {
        return $input->maxPool2d($this->kernelSize, $this->stride);
    }
}