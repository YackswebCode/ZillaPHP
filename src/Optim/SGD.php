<?php

declare(strict_types=1);

namespace ZillaPHP\Optim;

use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Tensor\Shape\Shape;

final class SGD extends Optimizer
{
    public function step(): void
    {
        foreach ($this->params as $p) {
            $g = $p->grad();
            if ($g === null) continue;

            $newData = [];
            $pd = $p->data();
            $gd = $g->data();
            foreach ($pd as $i => $v) $newData[] = $v - $this->lr * $gd[$i];

            // Replace parameter's data in place
            $this->setData($p, $newData);
        }
    }

    private function setData(\ZillaPHP\NN\Parameter $p, array $data): void
    {
        $ref = new \ReflectionClass(Tensor::class);
        $prop = $ref->getProperty('data');
        $prop->setAccessible(true);
        $prop->setValue($p, $data);
    }
}