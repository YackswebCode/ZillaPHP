<?php

declare(strict_types=1);

namespace ZillaPHP\Core;

use ZillaPHP\Hardware\CPU\CpuBackend;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;
use ZillaPHP\Tensor\Tensor;

final class Application
{
    public static function boot(bool $native = true, ?string $libraryPath = null): void
    {
        if ($native) {
            Tensor::setBackend(new NativeCpuBackend($libraryPath));
            return;
        }

        Tensor::setBackend(new CpuBackend());
    }
}