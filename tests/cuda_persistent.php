<?php
require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Hardware\CUDA\CudaBackend;
use ZillaPHP\Tensor\Tensor;

Application::boot(cuda: true);

$backend = Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required.\n");
    exit(1);
}

$N        = 1024;
$N_ITERS  = 100;
$nFloats  = $N * $N;

echo "=== Persistent device tensor benchmark ===\n";
echo "Matrix : {$N} x {$N}\n";
echo "Iters  : {$N_ITERS}\n\n";

// Generate inputs
mt_srand(42);
$a = []; $b = [];
for ($i = 0; $i < $nFloats; $i++) $a[] = mt_rand() / mt_getrandmax() - 0.5;
for ($i = 0; $i < $nFloats; $i++) $b[] = mt_rand() / mt_getrandmax() - 0.5;

// ---- Baseline: current slow path (upload+download every call) ----
echo "[1/2] Naive path — {toC → H2D → cublas → D2H → fromC} x {$N_ITERS}\n";

$tensorA = Tensor::fromArray($a)->reshape([$N, $N]);
$tensorB = Tensor::fromArray($b)->reshape([$N, $N]);

$backend->sync();
$t0 = microtime(true);
for ($i = 0; $i < $N_ITERS; $i++) {
    $tensorA->matmul($tensorB);
}
$backend->sync();
$naiveMs = (microtime(true) - $t0) * 1000;
printf("      total: %.1f ms   avg/call: %.2f ms\n\n", $naiveMs, $naiveMs / $N_ITERS);

// ---- Persistent path ----
echo "[2/2] Persistent path — upload once, GPU-resident, download once\n";

$aId = $backend->bufferAlloc($nFloats);
$bId = $backend->bufferAlloc($nFloats);
$cId = $backend->bufferAlloc($nFloats);

$t0 = microtime(true);
$backend->bufferUpload($aId, $a);
$backend->bufferUpload($bId, $b);
$backend->sync();
$uploadMs = (microtime(true) - $t0) * 1000;
printf("      upload:      %.1f ms\n", $uploadMs);

$backend->sync();
$t0 = microtime(true);
for ($i = 0; $i < $N_ITERS; $i++) {
    $backend->matmulDev($aId, $bId, $cId, $N, $N, $N);
}
$backend->sync();
$kernelMs = (microtime(true) - $t0) * 1000;
printf("      kernels:     %.1f ms   avg/call: %.2f ms\n", $kernelMs, $kernelMs / $N_ITERS);

$t0 = microtime(true);
$result = $backend->bufferDownload($cId, $nFloats);
$backend->sync();
$downloadMs = (microtime(true) - $t0) * 1000;
printf("      download:    %.1f ms\n", $downloadMs);

$totalPersistentMs = $uploadMs + $kernelMs + $downloadMs;
printf("      TOTAL:       %.1f ms\n\n", $totalPersistentMs);

$backend->bufferFree($aId);
$backend->bufferFree($bId);
$backend->bufferFree($cId);

// ---- Verdict ----
$speedup = $naiveMs / $totalPersistentMs;
printf("=== Result ===\n");
printf("Naive total:       %.1f ms\n", $naiveMs);
printf("Persistent total:  %.1f ms\n", $totalPersistentMs);
printf("Speedup:           %.1f x\n", $speedup);