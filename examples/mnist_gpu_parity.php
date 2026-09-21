<?php

declare(strict_types=1);

/**
 * Parity test: verify that the GPU forward path (matmul_dev) produces
 * the same results as a pure-PHP matmul reference.
 *
 * Uses small matrices so the test runs in under a second.
 */

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Hardware\CUDA\CudaBackend;
use ZillaPHP\Tensor\Tensor;

Application::boot(cuda: true);

$backend = Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required for this test.\n");
    exit(1);
}

// Small dimensions for a fast test
$N     = 8;
$IN    = 32;
$HID   = 16;

// Deterministic data
mt_srand(1234);

$w1 = [];
for ($i = 0; $i < $IN * $HID; $i++) {
    $w1[] = mt_rand() / mt_getrandmax() - 0.5;
}

$x = [];
for ($i = 0; $i < $N * $IN; $i++) {
    $x[] = mt_rand() / mt_getrandmax() - 0.5;
}

// ---- Reference: pure PHP forward ----
$z1Ref = [];
for ($i = 0; $i < $N; $i++) {
    $row = array_fill(0, $HID, 0.0);
    for ($k = 0; $k < $IN; $k++) {
        $xi = $x[$i * $IN + $k];
        if ($xi == 0.0) continue;
        for ($j = 0; $j < $HID; $j++) {
            $row[$j] += $xi * $w1[$k * $HID + $j];
        }
    }
    $z1Ref[] = $row;
}

// ---- GPU forward ----
$w1Gpu = $backend->bufferAlloc($IN * $HID);
$xGpu  = $backend->bufferAlloc($N * $IN);
$z1Gpu = $backend->bufferAlloc($N * $HID);

$backend->bufferUpload($w1Gpu, $w1);
$backend->bufferUpload($xGpu, $x);
$backend->matmulDev($xGpu, $w1Gpu, $z1Gpu, $N, $IN, $HID);
$z1GpuFlat = $backend->bufferDownload($z1Gpu, $N * $HID);
$backend->sync();

// ---- Compare ----
$maxErr = 0.0;
for ($i = 0; $i < $N; $i++) {
    for ($j = 0; $j < $HID; $j++) {
        $gpuV = $z1GpuFlat[$i * $HID + $j];
        $refV = $z1Ref[$i][$j];
        $maxErr = max($maxErr, abs($gpuV - $refV));
    }
}

printf("Shape:     %d x %d @ %d x %d\n", $N, $IN, $IN, $HID);
printf("Max error: %.10f\n", $maxErr);
echo ($maxErr < 1e-4) ? "PARITY OK\n" : "PARITY FAILED\n";

// ---- Cleanup ----
$backend->bufferFree($w1Gpu);
$backend->bufferFree($xGpu);
$backend->bufferFree($z1Gpu);

exit($maxErr < 1e-4 ? 0 : 1);