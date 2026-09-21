<?php

declare(strict_types=1);

namespace ZillaPHP\NN;

use ZillaPHP\Tensor\Tensor;

abstract class Module
{
    protected bool $training = true;

    abstract public function forward(Tensor $input): Tensor;

    public function __invoke(Tensor $input): Tensor
    {
        return $this->forward($input);
    }

    /** @return Tensor[] */
    public function parameters(): array
    {
        return [];
    }

    public function zeroGrad(): void
    {
        foreach ($this->parameters() as $p) {
            $p->zeroGrad();
        }
    }

    public function train(): void { $this->training = true; }
    public function eval(): void  { $this->training = false; }
    public function isTraining(): bool { return $this->training; }
}