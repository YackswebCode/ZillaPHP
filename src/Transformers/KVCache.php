<?php

declare(strict_types=1);

namespace ZillaPHP\Transformers;

use ZillaPHP\Tensor\Tensor;

final class KVCache
{
    /** @var array<int, array{k: Tensor, v: Tensor}> */
    public array $layers = [];

    public int $length = 0;

    public function __construct(private int $numLayers) {}

    public function numLayers(): int
    {
        return $this->numLayers;
    }

    public function reset(): void
    {
        $this->layers = [];
        $this->length = 0;
    }

    public function isEmpty(): bool
    {
        return $this->length === 0;
    }
}