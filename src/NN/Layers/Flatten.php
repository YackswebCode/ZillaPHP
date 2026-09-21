<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

final class Flatten extends Module
{
    public function forward(Tensor $input): Tensor
    {
        return $input->flatten();
    }
}