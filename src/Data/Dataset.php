<?php

declare(strict_types=1);

namespace ZillaPHP\Data;

interface Dataset
{
    public function size(): int;

    /** @return array{0: mixed, 1: mixed} */
    public function get(int $index): array;
}