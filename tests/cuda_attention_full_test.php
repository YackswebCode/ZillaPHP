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

// ---- Small fixed Q, K, V ----
$N = 4;         // sequence length
$D = 8;         // per-head dim
$scale = 1.0 / sqrt($D);

mt_srand(42);
$Q = randn($N * $D);
$K = randn($N * $D);
$V = randn($N * $D);

// ---- Reference (CPU) attention ----
$refOut = referenceAttention($Q, $K, $V, $N, $D, $scale);

// ---- GPU attention pipeline ----
$qId  = $backend->bufferAlloc($N * $D);
$kTId = $backend->bufferAlloc($D * $N);
$vId  = $backend->bufferAlloc($N * $D);
$sId  = $backend->bufferAlloc($N * $N);
$pId  = $backend->bufferAlloc($N * $N);
$oId  = $backend->bufferAlloc($N * $D);

// Upload Q
$backend->bufferUpload($qId, $Q);

// Upload K transposed [D, N] for matmul(Q, K^T)
$kT = [];
for ($d = 0; $d < $D; $d++) {
    for ($n = 0; $n < $N; $n++) $kT[] = $K[$n * $D + $d];
}
$backend->bufferUpload($kTId, $kT);

// Upload V
$backend->bufferUpload($vId, $V);

// S = Q @ K^T   [N, N]
$backend->matmulDev($qId, $kTId, $sId, $N, $D, $N);

// S *= scale
$backend->scaleDev($sId, $scale, $N * $N);

// S → causal softmax → P
$backend->causalSoftmaxDev($sId, $N, $N);
$backend->sync();
$sId2 = $backend->bufferDownload($sId, $N * $N);   // this is now P
$backend->bufferUpload($pId, $sId2);

// O = P @ V   [N, D]
$backend->matmulDev($pId, $vId, $oId, $N, $N, $D);
$backend->sync();
$got = $backend->bufferDownload($oId, $N * $D);

// ---- Compare ----
$maxErr = 0.0;
for ($i = 0; $i < $N * $D; $i++) {
    $maxErr = max($maxErr, abs($refOut[$i] - $got[$i]));
}

printf("Attention: N=%d, D=%d, scale=%.4f\n", $N, $D, $scale);
printf("Reference [0..3]: [%s]\n", implode(', ',
    array_map(fn($v) => sprintf('%.4f', $v), array_slice($refOut, 0, 4))));
printf("GPU       [0..3]: [%s]\n", implode(', ',
    array_map(fn($v) => sprintf('%.4f', $v), array_slice($got, 0, 4))));
printf("Max err: %.8f\n", $maxErr);
echo ($maxErr < 1e-4) ? "FULL ATTENTION OK\n" : "FULL ATTENTION FAILED\n";

// ---- Cleanup ----
foreach ([$qId, $kTId, $vId, $sId, $pId, $oId] as $id) $backend->bufferFree($id);

function randn(int $n): array {
    $o = [];
    for ($i = 0; $i < $n; $i++) {
        $u1 = mt_rand() / mt_getrandmax();
        $u2 = mt_rand() / mt_getrandmax();
        $o[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2);
    }
    return $o;
}

function referenceAttention(array $Q, array $K, array $V, int $N, int $D, float $scale): array {
    // S = Q @ K^T
    $S = [];
    for ($i = 0; $i < $N; $i++) {
        for ($j = 0; $j < $N; $j++) {
            $s = 0.0;
            for ($d = 0; $d < $D; $d++) $s += $Q[$i*$D+$d] * $K[$j*$D+$d];
            $S[$i*$N+$j] = $s * $scale;
        }
    }
    // Causal mask
    for ($i = 0; $i < $N; $i++)
        for ($j = 0; $j < $N; $j++)
            if ($j > $i) $S[$i*$N+$j] = -1e9;

    // Softmax per row
    $P = [];
    for ($i = 0; $i < $N; $i++) {
        $row = array_slice($S, $i * $N, $N);
        $mx = max($row);
        $sum = 0.0; $exps = [];
        foreach ($row as $v) { $e = exp($v - $mx); $exps[] = $e; $sum += $e; }
        foreach ($exps as $e) $P[] = $e / $sum;
    }

    // O = P @ V
    $O = [];
    for ($i = 0; $i < $N; $i++) {
        for ($d = 0; $d < $D; $d++) {
            $o = 0.0;
            for ($j = 0; $j < $N; $j++) $o += $P[$i*$N+$j] * $V[$j*$D+$d];
            $O[$i*$D+$d] = $o;
        }
    }
    return $O;
}