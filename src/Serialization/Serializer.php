<?php

declare(strict_types=1);

namespace ZillaPHP\Serialization;

interface Serializer
{
    /** @param array<string, array{shape:int[], data:(int|float|bool)[]}> $state */
    public function save(array $state, string $path): void;

    /** @return array<string, array{shape:int[], data:(int|float|bool)[]}> */
    public function load(string $path): array;
}