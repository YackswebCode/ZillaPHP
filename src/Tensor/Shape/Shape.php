<?php

declare(strict_types=1);

namespace ZillaPHP\Tensor\Shape;

final class Shape
{
    /** @var int[] */
    private array $dims;

    /** @param int[] $dims */
    public function __construct(array $dims)
    {
        foreach ($dims as $d) {
            if (!is_int($d) || $d < 0) {
                throw new \InvalidArgumentException("Shape dimensions must be non-negative integers.");
            }
        }
        $this->dims = array_values($dims);
    }

    /** @return int[] */
    public function dims(): array
    {
        return $this->dims;
    }

    public function ndim(): int
    {
        return count($this->dims);
    }

    public function size(): int
    {
        if (empty($this->dims)) {
            return 0;
        }
        return array_product($this->dims);
    }

    /** @return int[] */
    public function strides(): array
    {
        $strides = [];
        $acc = 1;
        for ($i = $this->ndim() - 1; $i >= 0; $i--) {
            $strides[$i] = $acc;
            $acc *= $this->dims[$i];
        }
        return $strides;
    }

    public function __toString(): string
    {
        return '[' . implode(', ', $this->dims) . ']';
    }
}