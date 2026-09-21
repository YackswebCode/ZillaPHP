<?php

declare(strict_types=1);

namespace ZillaPHP\Data;

/**
 * Iterates a Dataset in batches. For now, batches are materialized PHP arrays
 * of [Tensor, target] pairs. Native/graph execution and prefetching come later.
 */
final class DataLoader implements \IteratorAggregate
{
    public function __construct(
        private Dataset $dataset,
        private int $batchSize = 32,
        private bool $shuffle = true,
    ) {
        if ($batchSize < 1) throw new \InvalidArgumentException("batchSize must be >= 1.");
    }

    /** @return \Generator<array<array{0:mixed, 1:mixed}>> */
    public function getIterator(): \Generator
    {
        $n = $this->dataset->size();
        $indices = range(0, $n - 1);
        if ($this->shuffle) {
            shuffle($indices);
        }

        for ($i = 0; $i < $n; $i += $this->batchSize) {
            $batch = [];
            for ($j = $i; $j < min($i + $this->batchSize, $n); $j++) {
                $batch[] = $this->dataset->get($indices[$j]);
            }
            yield $batch;
        }
    }

    public function size(): int { return $this->dataset->size(); }
    public function batchSize(): int { return $this->batchSize; }
}