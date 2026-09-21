<?php

declare(strict_types=1);

namespace ZillaPHP\Optim;

use ZillaPHP\NN\Parameter;

abstract class Optimizer
{
    /** @var Parameter[] */
    protected array $params;
    protected float $lr;

    /** @param Parameter[] $params */
    public function __construct(array $params, float $lr = 0.01)
    {
        $this->params = $params;
        $this->lr = $lr;
    }

    public function zeroGrad(): void
    {
        foreach ($this->params as $p) $p->zeroGrad();
    }

    abstract public function step(): void;
}