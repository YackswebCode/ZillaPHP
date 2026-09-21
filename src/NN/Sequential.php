<?php

declare(strict_types=1);

namespace ZillaPHP\NN;

use ZillaPHP\Tensor\Tensor;

final class Sequential extends Module
{
    /** @var Module[] */
    private array $layers;

    /** @param Module[] $layers */
    public function __construct(array $layers)
    {
        $this->layers = $layers;
    }

    public function forward(Tensor $input): Tensor
    {
        $x = $input;
        foreach ($this->layers as $layer) {
            $x = $layer->forward($x);
        }
        return $x;
    }

    /** @return Module[] */
    public function layers(): array { return $this->layers; }

    public function parameters(): array
    {
        $params = [];
        foreach ($this->layers as $layer) {
            foreach ($layer->parameters() as $p) $params[] = $p;
        }
        return $params;
    }

    public function train(): void
    {
        parent::train();
        foreach ($this->layers as $l) $l->train();
    }
    public function eval(): void
    {
        parent::eval();
        foreach ($this->layers as $l) $l->eval();
    }
}