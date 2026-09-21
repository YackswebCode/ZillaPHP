<?php

declare(strict_types=1);

namespace ZillaPHP\Tensor\DType;

enum DType: string
{
    case FLOAT32 = 'float32';
    case FLOAT64 = 'float64';
    case INT32   = 'int32';
    case INT64   = 'int64';
    case BOOL    = 'bool';

    public function bytes(): int
    {
        return match ($this) {
            self::FLOAT32 => 4,
            self::FLOAT64 => 8,
            self::INT32   => 4,
            self::INT64   => 8,
            self::BOOL    => 1,
        };
    }
}