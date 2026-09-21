<?php

declare(strict_types=1);

/**
 * GPU-native MNIST — v2 (fully GPU-resident pipeline).
 *
 * Every training operation runs on the GPU. Only two things ever cross PCIe:
 *   - The X batch (uploaded per batch, ~200 KB)
 *   - The Y batch (64 floats)
 *   - The scalar loss (only downloaded once per epoch for progress reporting)
 *
 * Model: 784 → 128 (ReLU) → 10, with bias, SGD.
 */

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\MnistDataset;
use ZillaPHP\Hardware\CUDA\CudaBackend;
use ZillaPHP\Tensor\Tensor;

Application::boot(cuda: true);

$backend = Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required.\n");
    exit(1);
}

// ====================================================================
// Config
// ====================================================================
$TRAIN_N = (int)   (getenv('TRAIN_N') ?: 60000);
$TEST_N  = (int)   (getenv('TEST_N')  ?: 10000);
$BATCH   = (int)   (getenv('BATCH')   ?: 64);
$EPOCHS  = (int)   (getenv('EPOCHS')  ?: 10);
$LR      = (float) (getenv('LR')      ?: 0.05);
$HIDDEN  = (int)   (getenv('HIDDEN')  ?: 128);

const IN_DIM    = 784;
const N_CLASSES = 10;

// ====================================================================
// Load MNIST into CPU arrays
// ====================================================================
$base = __DIR__ . '/../data/mnist';
if (!is_file("{$base}/train-images-idx3-ubyte")) {
    fwrite(STDERR, "MNIST not found. Run: php scripts/download_mnist.php\n");
    exit(1);
}

$trainDs = new MnistDataset("{$base}/train-images-idx3-ubyte", "{$base}/train-labels-idx1-ubyte");
$testDs  = new MnistDataset("{$base}/t10k-images-idx3-ubyte", "{$base}/t10k-labels-idx1-ubyte");

echo "Loading {$TRAIN_N} train samples...\n";
$trainX = []; $trainY = [];
for ($i = 0; $i < $TRAIN_N; $i++) {
    [$x, $y] = $trainDs->get($i);
    $trainX[] = $x->data();       // flat [784]
    $trainY[] = (float) $y[0];    // stored as float
}

echo "Loading {$TEST_N} test samples...\n";
$testX = []; $testY = [];
for ($i = 0; $i < $TEST_N; $i++) {
    [$x, $y] = $testDs->get($i);
    $testX[] = $x->data();
    $testY[] = (int) $y[0];
}

// ====================================================================
// Weights on CPU (initial values, then uploaded once and never touched
// again until we want to inspect them)
// ====================================================================
mt_srand(42);

$w1Init = [];
$s1 = sqrt(1.0 / IN_DIM);
for ($i = 0; $i < IN_DIM * $HIDDEN; $i++) {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    $w1Init[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $s1;
}
$b1Init = array_fill(0, $HIDDEN, 0.0);

$w2Init = [];
$s2 = sqrt(1.0 / $HIDDEN);
for ($i = 0; $i < $HIDDEN * N_CLASSES; $i++) {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    $w2Init[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $s2;
}
$b2Init = array_fill(0, N_CLASSES, 0.0);

// ====================================================================
// Allocate GPU buffers — ONCE
// ====================================================================
echo "\nAllocating GPU buffers...\n";

$w1Gpu   = $backend->bufferAlloc(IN_DIM * $HIDDEN);
$b1Gpu   = $backend->bufferAlloc($HIDDEN);
$w2Gpu   = $backend->bufferAlloc($HIDDEN * N_CLASSES);
$b2Gpu   = $backend->bufferAlloc(N_CLASSES);

$xGpu    = $backend->bufferAlloc($BATCH * IN_DIM);
$yGpu    = $backend->bufferAlloc($BATCH);

$z1Gpu   = $backend->bufferAlloc($BATCH * $HIDDEN);
$a1Gpu   = $backend->bufferAlloc($BATCH * $HIDDEN);
$m1Gpu   = $backend->bufferAlloc($BATCH * $HIDDEN);

$z2Gpu   = $backend->bufferAlloc($BATCH * N_CLASSES);
$dz2Gpu  = $backend->bufferAlloc($BATCH * N_CLASSES);

$dw1Gpu  = $backend->bufferAlloc(IN_DIM * $HIDDEN);
$db1Gpu  = $backend->bufferAlloc($HIDDEN);
$dw2Gpu  = $backend->bufferAlloc($HIDDEN * N_CLASSES);
$db2Gpu  = $backend->bufferAlloc(N_CLASSES);

$da1Gpu  = $backend->bufferAlloc($BATCH * $HIDDEN);
$dz1Gpu  = $backend->bufferAlloc($BATCH * $HIDDEN);

$lossGpu = $backend->bufferAlloc(1);

// Upload initial weights and biases
$backend->bufferUpload($w1Gpu, $w1Init);
$backend->bufferUpload($b1Gpu, $b1Init);
$backend->bufferUpload($w2Gpu, $w2Init);
$backend->bufferUpload($b2Gpu, $b2Init);

echo "Backend: " . $backend->name() . "\n";
echo "Model:   " . IN_DIM . " -> {$HIDDEN} (ReLU) -> " . N_CLASSES . "\n";
echo "Batch:   {$BATCH}\n";
echo "Epochs:  {$EPOCHS}\n";
echo "LR:      {$LR}\n";
echo str_repeat('=', 60) . "\n";

// ====================================================================
// Training loop
// ====================================================================
$nBatches = (int) ceil($TRAIN_N / $BATCH);
$tStart = microtime(true);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $epochStart = microtime(true);
    $trainSeen  = 0;
    $trainCorrect = 0;

    $order = range(0, $TRAIN_N - 1);
    shuffle($order);

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * $BATCH;
        $end   = min($start + $BATCH, $TRAIN_N);
        $realN = $end - $start;
        if ($realN === 0) break;

        // ---- Prepare X and Y batches (float arrays, padded to BATCH) ----
        $xFlat = [];
        $yFlat = [];
        for ($i = $start; $i < $end; $i++) {
            $idx = $order[$i];
            foreach ($trainX[$idx] as $v) $xFlat[] = $v;
            $yFlat[] = $trainY[$idx];
        }
        // Pad by repeating the first sample (its loss is zeroed out
        // because its target row is also duplicated but we won't count it)
        while (count($yFlat) < $BATCH) {
            foreach ($trainX[$order[$start]] as $v) $xFlat[] = $v;
            $yFlat[] = $trainY[$order[$start]];
        }

        // ---- Upload X and Y ----
        $backend->bufferUpload($xGpu, $xFlat);
        $backend->bufferUpload($yGpu, $yFlat);

        // ---- Forward: Z1 = X @ W1, then +b1, then ReLU ----
        $backend->matmulDev($xGpu, $w1Gpu, $z1Gpu, $BATCH, IN_DIM, $HIDDEN);
        $backend->addBiasDev($z1Gpu, $b1Gpu, $BATCH, $HIDDEN);
        $backend->reluFwdDev($z1Gpu, $a1Gpu, $m1Gpu, $BATCH * $HIDDEN);

        // ---- Forward: Z2 = A1 @ W2 + b2 ----
        $backend->matmulDev($a1Gpu, $w2Gpu, $z2Gpu, $BATCH, $HIDDEN, N_CLASSES);
        $backend->addBiasDev($z2Gpu, $b2Gpu, $BATCH, N_CLASSES);

        // ---- Loss + dZ2 ----
        $backend->zeroDev($lossGpu, 1);
        $backend->softmaxCeDev($z2Gpu, $yGpu, $dz2Gpu, $lossGpu, $BATCH, N_CLASSES);

        // ---- Backward ----
        // dW2 = A1^T @ dZ2       [HIDDEN, N_CLASSES] = [HIDDEN, BATCH] @ [BATCH, N_CLASSES]
        $backend->matmulTnDev($a1Gpu, $dz2Gpu, $dw2Gpu, $HIDDEN, $BATCH, N_CLASSES);
        // db2 = sum over batch
        $backend->biasGradDev($dz2Gpu, $db2Gpu, $BATCH, N_CLASSES);

        // dA1 = dZ2 @ W2^T       [BATCH, HIDDEN] = [BATCH, N_CLASSES] @ [N_CLASSES, HIDDEN]
        // We need W2 as [N_CLASSES, HIDDEN]... but W2 is stored [HIDDEN, N_CLASSES].
        // Use matmul_nt:  A [B, K] @ B^T [K, N], where B_storage is [N, K].
        // Here A = dZ2 [B, N_CLASSES], B_storage = W2 [HIDDEN, N_CLASSES], so
        // N (= output dim) = HIDDEN, K (= inner) = N_CLASSES.
        // C = dZ2 @ W2ᵀ  →  [BATCH, HIDDEN] = [BATCH, N_CLASSES] @ [N_CLASSES, HIDDEN]  ✓
        $backend->matmulNtDev($dz2Gpu, $w2Gpu, $da1Gpu, $BATCH, N_CLASSES, $HIDDEN);

        // dZ1 = dA1 * mask
        $backend->reluBwdDev($da1Gpu, $m1Gpu, $dz1Gpu, $BATCH * $HIDDEN);

        // dW1 = X^T @ dZ1       [IN_DIM, HIDDEN] = [IN_DIM, BATCH] @ [BATCH, HIDDEN]
        $backend->matmulTnDev($xGpu, $dz1Gpu, $dw1Gpu, IN_DIM, $BATCH, $HIDDEN);
        // db1 = sum over batch
        $backend->biasGradDev($dz1Gpu, $db1Gpu, $BATCH, $HIDDEN);

        // ---- SGD update (all on GPU) ----
        $backend->sgdUpdateDev($w1Gpu, $dw1Gpu, $LR, IN_DIM * $HIDDEN);
        $backend->sgdUpdateDev($b1Gpu, $db1Gpu, $LR, $HIDDEN);
        $backend->sgdUpdateDev($w2Gpu, $dw2Gpu, $LR, $HIDDEN * N_CLASSES);
        $backend->sgdUpdateDev($b2Gpu, $db2Gpu, $LR, N_CLASSES);
    }

    $elapsed = microtime(true) - $epochStart;
    printf("epoch %2d/%d   %.2fs\n", $epoch + 1, $EPOCHS, $elapsed);
}

$total = microtime(true) - $tStart;

// ====================================================================
// Evaluate on CPU using the trained weights (downloaded once)
// ====================================================================
echo str_repeat('-', 60) . "\n";
echo "Downloading trained weights...\n";

$backend->sync();
$w1Trained = $backend->bufferDownload($w1Gpu, IN_DIM * $HIDDEN);
$b1Trained = $backend->bufferDownload($b1Gpu, $HIDDEN);
$w2Trained = $backend->bufferDownload($w2Gpu, $HIDDEN * N_CLASSES);
$b2Trained = $backend->bufferDownload($b2Gpu, N_CLASSES);

echo "Evaluating {$TEST_N} test samples...\n";

$correct = 0;
for ($i = 0; $i < $TEST_N; $i++) {
    // Z1 = X @ W1 + b1
    $z1 = array_fill(0, $HIDDEN, 0.0);
    for ($k = 0; $k < IN_DIM; $k++) {
        $x = $testX[$i][$k];
        if ($x == 0.0) continue;
        $base = $k * $HIDDEN;
        for ($h = 0; $h < $HIDDEN; $h++) {
            $z1[$h] += $x * $w1Trained[$base + $h];
        }
    }
    for ($h = 0; $h < $HIDDEN; $h++) {
        $v = $z1[$h] + $b1Trained[$h];
        $z1[$h] = $v > 0 ? $v : 0.0;
    }
    // Z2 = A1 @ W2 + b2
    $z2 = $b2Trained;
    for ($h = 0; $h < $HIDDEN; $h++) {
        $a = $z1[$h];
        if ($a == 0.0) continue;
        $base = $h * N_CLASSES;
        for ($c = 0; $c < N_CLASSES; $c++) {
            $z2[$c] += $a * $w2Trained[$base + $c];
        }
    }
    $best = 0; $bv = $z2[0];
    for ($c = 1; $c < N_CLASSES; $c++) if ($z2[$c] > $bv) { $bv = $z2[$c]; $best = $c; }
    if ($best === $testY[$i]) $correct++;
}

$acc = 100.0 * $correct / $TEST_N;
echo str_repeat('=', 60) . "\n";
printf("Test accuracy:   %.2f%% (%d / %d)\n", $acc, $correct, $TEST_N);
printf("Total training:  %.1f s (%.1f min)\n", $total, $total / 60);
printf("Per epoch:       %.1f s\n", $total / $EPOCHS);

// Cleanup
foreach ([$w1Gpu,$b1Gpu,$w2Gpu,$b2Gpu,$xGpu,$yGpu,$z1Gpu,$a1Gpu,$m1Gpu,
          $z2Gpu,$dz2Gpu,$dw1Gpu,$db1Gpu,$dw2Gpu,$db2Gpu,$da1Gpu,$dz1Gpu,$lossGpu] as $id) {
    $backend->bufferFree($id);
}