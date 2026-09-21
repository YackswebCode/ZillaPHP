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
        // Default OpenBLAS to single-threaded. Multi-threaded BLAS adds
        // ~1ms of thread-sync overhead per call, which loses to the naive
        // vectorized loop for the matrix sizes we use in practice.
        //
        // Users can override with ZILLA_OPENBLAS_THREADS=N if they want
        // to experiment with multi-threaded BLAS.
        //
        // Must be set BEFORE FFI dlopens libzilla_cpu.so.
        $threads = getenv('ZILLA_OPENBLAS_THREADS');
        if ($threads === false) {
            $threads = '1';
        }
        putenv('OPENBLAS_NUM_THREADS=' . $threads);

        if ($native) {
            Tensor::setBackend(new NativeCpuBackend($libraryPath));
            return;
        }

        Tensor::setBackend(new CpuBackend());
    }
}