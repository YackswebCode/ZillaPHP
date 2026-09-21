<?php
require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Tensor\Shape\Shape;

function bench(callable $fn, int $iters = 5): float {
    $fn(); $fn();  // warmup
    $t0 = microtime(true);
    for ($i = 0; $i < $iters; $i++) $fn();
    return (microtime(true) - $t0) / $iters * 1000;
}

$sizes = [128, 256, 512, 1024];

printf("%-16s %12s %12s %10s\n", 'Shape', 'CPU (ms)', 'CUDA (ms)', 'Speedup');
echo str_repeat('-', 54) . "\n";

foreach ($sizes as $n) {
    mt_srand(42);
    $aData = []; $bData = [];
    for ($i = 0; $i < $n * $n; $i++) $aData[] = mt_rand() / mt_getrandmax() - 0.5;
    for ($i = 0; $i < $n * $n; $i++) $bData[] = mt_rand() / mt_getrandmax() - 0.5;

    Application::boot(native: true);
    $a = new Tensor($aData, new Shape([$n, $n]));
    $b = new Tensor($bData, new Shape([$n, $n]));
    $cpuMs = bench(fn() => $a->matmul($b));

    Application::boot(cuda: true);
    $a = new Tensor($aData, new Shape([$n, $n]));
    $b = new Tensor($bData, new Shape([$n, $n]));
    $gpuMs = bench(fn() => $a->matmul($b));

    printf("%4dx%4dx%4d    %10.2f  %10.2f    %6.2fx\n",
        $n, $n, $n, $cpuMs, $gpuMs, $cpuMs / max($gpuMs, 0.001));
}