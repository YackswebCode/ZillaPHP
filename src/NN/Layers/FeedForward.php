<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

final class FeedForward extends Module
{
    private Linear $fc1;
    private Linear $fc2;
    private ReLU   $relu;

    public function __construct(int $dim, int $hidden)
    {
        $this->fc1  = new Linear($dim, $hidden);
        $this->fc2  = new Linear($hidden, $dim);
        $this->relu = new ReLU();
    }

    public function forward(Tensor $x): Tensor
    {
        return $this->fc2->forward($this->relu->forward($this->fc1->forward($x)));
    }

    /** @return \ZillaPHP\NN\Parameter[] */
    public function parameters(): array
    {
        return array_merge($this->fc1->parameters(), $this->fc2->parameters());
    }
}