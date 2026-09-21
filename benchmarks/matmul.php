<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

function bench(callable $fn): float
{
    $start = microtime(true);
    $fn();
    return microtime(true) - $start;
}

$sizes = [
    [64, 64, 64],
    [128, 128, 128],
    [256, 256, 256],
    [784, 128, 128],
    [512, 512, 512],
];

printf("%-14s %-14s %-14s %-14s %-10s\n", 'A', 'B', 'pure PHP', 'native C', 'speedup');
printf("%s\n", str_repeat('-', 70));

foreach ($sizes as [$m, $k, $n]) {
    $a = Tensor::randn([$m, $k]);
    $b = Tensor::randn([$k, $n]);

    Application::boot(native: false);
    $phpTime = bench(fn() => $a->matmul($b));

    Application::boot(native: true);
    $nativeTime = bench(fn() => $a->matmul($b));

    $speedup = $nativeTime > 0 ? $phpTime / $nativeTime : 0;

    printf("%-14s %-14s %-14s %-14s %.1fx\n",
        "{$m}x{$k}",
        "{$k}x{$n}",
        sprintf('%.2f ms', $phpTime * 1000),
        sprintf('%.2f ms', $nativeTime * 1000),
        $speedup,
    );
}

Application::boot(native: true);
$backend = Tensor::backend();
printf("\nActive backend : %s\n", $backend->name());
if (method_exists($backend, 'isNative')) {
    printf("Native FFI    : %s\n", $backend->isNative() ? 'yes' : 'no');
}