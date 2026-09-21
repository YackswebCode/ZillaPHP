<?php

declare(strict_types=1);

namespace ZillaPHP\Data;

final class ArrayDataset implements Dataset
{
    /** @param array<array{0:mixed, 1:mixed}> $items */
    public function __construct(private array $items) {}

    public function size(): int
    {
        return count($this->items);
    }

    public function get(int $index): array
    {
        if (!isset($this->items[$index])) {
            throw new \OutOfRangeException("Index {$index} out of range.");
        }
        return $this->items[$index];
    }
}