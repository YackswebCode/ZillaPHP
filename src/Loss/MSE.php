<?php

declare(strict_types=1);

namespace ZillaPHP\Loss;

use ZillaPHP\Tensor\Tensor;

final class MSE extends Loss
{
    public function forward(Tensor $prediction, Tensor|array $target): Tensor
    {
        $targetT = $target instanceof Tensor ? $target : Tensor::fromArray($target);
        $diff    = $prediction->sub($targetT);
        $squared = $diff->mul($diff);
        return $squared->mean();
    }
}