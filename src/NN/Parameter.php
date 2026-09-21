<?php

declare(strict_types=1);

namespace ZillaPHP\NN;

use ZillaPHP\Tensor\DType\DType;
use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

/**
 * A Parameter is a Tensor whose values are intended to be learned.
 *
 * It always requires gradients, which is what makes it register with
 * the autograd engine when used in a forward pass.
 */
final class Parameter extends Tensor
{
    /** @param array<int|float|bool> $data */
    public function __construct(array $data, Shape $shape)
    {
        parent::__construct($data, $shape, DType::FLOAT32, null, true);
    }
}