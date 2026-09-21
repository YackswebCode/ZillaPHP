<?php

declare(strict_types=1);

namespace ZillaPHP\Core;

use ZillaPHP\Hardware\CPU\CpuBackend;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;
use ZillaPHP\Hardware\CUDA\CudaBackend;
use ZillaPHP\Tensor\Tensor;

final class Application
{
    public static function boot(
        bool $native = true,
        ?string $libraryPath = null,
        bool $cuda = false,
        ?string $cudaLibraryPath = null,
    ): void {
        // Default OpenBLAS to single-threaded (see benchmarks/matmul.php).
        $threads = getenv('ZILLA_OPENBLAS_THREADS');
        if ($threads === false) $threads = '1';
        putenv('OPENBLAS_NUM_THREADS=' . $threads);

        // CUDA takes priority if requested and available.
        if ($cuda) {
            $cudaBackend = new CudaBackend($cudaLibraryPath);
            if ($cudaBackend->isCuda()) {
                Tensor::setBackend($cudaBackend);
                return;
            }
            // fall through to CPU if CUDA not available
        }

        if ($native) {
            Tensor::setBackend(new NativeCpuBackend($libraryPath));
            return;
        }

        Tensor::setBackend(new CpuBackend());
    }
}