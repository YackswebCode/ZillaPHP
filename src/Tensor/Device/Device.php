<?php

declare(strict_types=1);

namespace ZillaPHP\Tensor\Device;

final class Device
{
    private function __construct(
        private readonly string $type,
        private readonly int $index = 0,
    ) {}

    public static function cpu(): self
    {
        return new self('cpu');
    }

    public static function cuda(int $index = 0): self
    {
        return new self('cuda', $index);
    }

    public static function rocm(int $index = 0): self
    {
        return new self('rocm', $index);
    }

    public function type(): string
    {
        return $this->type;
    }

    public function index(): int
    {
        return $this->index;
    }

    public function isCpu(): bool
    {
        return $this->type === 'cpu';
    }

    public function __toString(): string
    {
        return $this->type . ':' . $this->index;
    }
}