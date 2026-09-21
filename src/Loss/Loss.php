<?php

declare(strict_types=1);

namespace ZillaPHP\Loss;

use ZillaPHP\Tensor\Tensor;

abstract class Loss
{
    abstract public function forward(Tensor $prediction, Tensor|array $target): Tensor;
}