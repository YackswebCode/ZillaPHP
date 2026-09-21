<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Activations;

use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

final class ReLU extends Module
{
    public function forward(Tensor $input): Tensor
    {
        return $input->relu();
    }
}