<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

// Detect environment
$hasCuda = is_file(__DIR__ . '/../native/cuda/libzilla_cuda.so');

echo "CUDA library present: " . ($hasCuda ? 'yes' : 'no') . "\n";
echo "Running benchmarks...\n\n";

$sizes = [
    [128, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [1024, 1024, 1024],
];

printf("%-14s %-14s %-14s %-14s\n", 'A', 'B', 'CPU+native', 'CUDA');
printf("%s\n", str_repeat('-', 60));

foreach ($sizes as [$m, $k, $n]) {
    $a = Tensor::randn([$m, $k]);
    $b = Tensor::randn([$k, $n]);

    // CPU
    Application::boot(native: true);
    $t0 = microtime(true);
    $a->matmul($b);
    $cpuMs = (microtime(true) - $t0) * 1000;

    // CUDA
    $cudaMs = null;
    if ($hasCuda) {
        Application::boot(cuda: true);
        if (Tensor::backend()->name() === 'cuda') {
            $t0 = microtime(true);
            $a->matmul($b);
            $cudaMs = (microtime(true) - $t0) * 1000;
        }
    }

    printf(
        "%-14s %-14s %-14s %-14s\n",
        "{$m}x{$k}",
        "{$k}x{$n}",
        sprintf('%.2f ms', $cpuMs),
        $cudaMs === null ? 'N/A' : sprintf('%.2f ms', $cudaMs),
    );
}