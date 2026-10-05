<?php

declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Hardware\CUDA\CudaBackend;

Application::boot(cuda: true);

$backend = \ZillaPHP\Tensor\Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required.\n");
    exit(1);
}

// =============================================================
// Test 1: plain softmax over rows [4, 5]
// =============================================================
$N = 4; $M = 5;
$scores = [
    1.0, 2.0, 3.0, 4.0, 5.0,
    2.0, 2.0, 2.0, 2.0, 2.0,
    -1.0, 0.0, 1.0, -1.0, 0.0,
    100.0, 200.0, 300.0, 400.0, 500.0,
];

// Reference softmax
$refSoftmax = [];
for ($i = 0; $i < $N; $i++) {
    $row = array_slice($scores, $i * $M, $M);
    $mx = max($row);
    $sum = 0.0;
    $exps = [];
    foreach ($row as $v) { $e = exp($v - $mx); $exps[] = $e; $sum += $e; }
    foreach ($exps as $e) $refSoftmax[] = $e / $sum;
}

$xId = $backend->bufferAlloc($N * $M);
$yId = $backend->bufferAlloc($N * $M);
$backend->bufferUpload($xId, $scores);

$backend->softmaxRowsFwdDev($xId, $yId, $N, $M);
$backend->sync();
$got = $backend->bufferDownload($yId, $N * $M);

$maxErr = 0.0;
foreach ($refSoftmax as $i => $v) $maxErr = max($maxErr, abs($v - $got[$i]));

printf("=== Test 1: plain softmax [%d, %d] ===\n", $N, $M);
printf("Max err:  %.8f\n", $maxErr);
echo ($maxErr < 1e-5) ? "  OK\n\n" : "  FAILED\n\n";

// =============================================================
// Test 2: causal mask + softmax
// =============================================================
$N = 4; $M = 4;
$scores = [
    1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0,
    1.0, 2.0, 3.0, 4.0,
];

// Reference causal softmax
$refCausal = [];
for ($i = 0; $i < $N; $i++) {
    $row = array_slice($scores, $i * $M, $M);
    // Apply mask
    for ($j = 0; $j < $M; $j++) if ($j > $i) $row[$j] = -1e9;
    // Softmax
    $mx = max($row);
    $sum = 0.0; $exps = [];
    foreach ($row as $v) { $e = exp($v - $mx); $exps[] = $e; $sum += $e; }
    foreach ($exps as $e) $refCausal[] = $e / $sum;
}

$zId = $backend->bufferAlloc($N * $M);
$backend->bufferUpload($zId, $scores);
$backend->causalSoftmaxDev($zId, $N, $M);
$backend->sync();
$got = $backend->bufferDownload($zId, $N * $M);

$maxErr = 0.0;
foreach ($refCausal as $i => $v) $maxErr = max($maxErr, abs($v - $got[$i]));

printf("=== Test 2: causal softmax [%d, %d] ===\n", $N, $M);
printf("Got rows:\n");
for ($i = 0; $i < $N; $i++) {
    printf("  [%s]\n", implode(', ', array_map(fn($v) => sprintf('%.4f', $v), array_slice($got, $i * $M, $M))));
}
printf("Max err:  %.8f\n", $maxErr);
echo ($maxErr < 1e-5) ? "  OK\n\n" : "  FAILED\n\n";

// =============================================================
// Test 3: scale
// =============================================================
$n = 6;
$x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
$sId = $backend->bufferAlloc($n);
$backend->bufferUpload($sId, $x);
$backend->scaleDev($sId, 0.5, $n);
$backend->sync();
$got = $backend->bufferDownload($sId, $n);

$expected = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
$maxErr = 0.0;
foreach ($expected as $i => $v) $maxErr = max($maxErr, abs($v - $got[$i]));

printf("=== Test 3: scale by 0.5 ===\n");
printf("Expected: %s\n", json_encode($expected));
printf("Got:      %s\n", json_encode($got));
printf("Max err:  %.8f\n", $maxErr);
echo ($maxErr < 1e-6) ? "  OK\n\n" : "  FAILED\n\n";

// =============================================================
// Test 4: softmax backward
// =============================================================
$N = 2; $M = 3;
$P = [
    0.2, 0.3, 0.5,
    0.1, 0.8, 0.1,
];
$dP = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
];

// Reference: dS[i,j] = P[i,j] * (dP[i,j] - sum_k dP[i,k]*P[i,k])
$refDS = [];
for ($i = 0; $i < $N; $i++) {
    $dot = 0.0;
    for ($j = 0; $j < $M; $j++) $dot += $dP[$i*$M+$j] * $P[$i*$M+$j];
    for ($j = 0; $j < $M; $j++) {
        $refDS[] = $P[$i*$M+$j] * ($dP[$i*$M+$j] - $dot);
    }
}

$pId  = $backend->bufferAlloc($N * $M);
$dpId = $backend->bufferAlloc($N * $M);
$dsId = $backend->bufferAlloc($N * $M);
$backend->bufferUpload($pId, $P);
$backend->bufferUpload($dpId, $dP);
$backend->softmaxRowsBwdDev($pId, $dpId, $dsId, $N, $M);
$backend->sync();
$got = $backend->bufferDownload($dsId, $N * $M);

$maxErr = 0.0;
foreach ($refDS as $i => $v) $maxErr = max($maxErr, abs($v - $got[$i]));

printf("=== Test 4: softmax backward [%d, %d] ===\n", $N, $M);
printf("Expected: %s\n", json_encode($refDS));
printf("Got:      %s\n", json_encode($got));
printf("Max err:  %.8f\n", $maxErr);
echo ($maxErr < 1e-5) ? "  OK\n" : "  FAILED\n";