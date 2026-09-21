<?php

declare(strict_types=1);

/**
 * GPU-native MNIST training.
 *
 * Uses persistent device buffers to keep W1 on the GPU across the whole
 * run. Only the per-batch data and the transposed gradients move over PCIe.
 *
 * Design (v1 — deliberate simplifications):
 *
 *   - No bias. Bias costs a new GPU kernel and complicates the backward
 *     pass through X_aug. We accept ~1-2% accuracy loss for a working demo.
 *     Adding bias later is a small change (augment X with a ones column).
 *
 *   - Plain SGD (no momentum, no Adam). Adam needs per-parameter state
 *     buffers on the GPU. SGD only needs dW1 → W1 -= lr * dW1.
 *
 *   - Layer 2 (Z2 = A1 @ W2) runs on CPU. It's 128x10 — negligible.
 *
 *   - Layer 1 (Z1 = X @ W1) and its backward (dW1 = X^T @ dZ1) run on GPU
 *     via cuBLAS — these are the operations that dominate the compute.
 *
 * The two matmuls on GPU are:
 *
 *   Z1    = X @ W1           [BATCH,784] @ [784,128] -> [BATCH,128]
 *   dW1_T = dZ1_T @ X        [128,BATCH] @ [BATCH,784] -> [128,784]
 *
 * We compute dW1 in transposed form because cuBLAS gives better performance
 * when the output rows are contiguous, and because transposing dZ1 on CPU
 * (small) is cheaper than transposing dW1 (large) on CPU afterward.
 */

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\MnistDataset;
use ZillaPHP\Hardware\CUDA\CudaBackend;
use ZillaPHP\Tensor\Tensor;

Application::boot(cuda: true);

$backend = Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required. Build native/cuda/libzilla_cuda.so first.\n");
    exit(1);
}

// ====================================================================
// Configuration
// ====================================================================
$TRAIN_N = (int)   (getenv('TRAIN_N') ?: 60000);
$TEST_N  = (int)   (getenv('TEST_N')  ?: 10000);
$BATCH   = (int)   (getenv('BATCH')   ?: 64);
$EPOCHS  = (int)   (getenv('EPOCHS')  ?: 10);
$LR      = (float) (getenv('LR')      ?: 0.01);
$HIDDEN  = (int)   (getenv('HIDDEN')  ?: 128);

const IN_DIM    = 784;
const N_CLASSES = 10;

// ====================================================================
// Load MNIST
// ====================================================================
$base = __DIR__ . '/../data/mnist';

if (!is_file("{$base}/train-images-idx3-ubyte")) {
    fwrite(STDERR, "MNIST not found. Run: php scripts/download_mnist.php\n");
    exit(1);
}

$trainDs = new MnistDataset(
    "{$base}/train-images-idx3-ubyte",
    "{$base}/train-labels-idx1-ubyte"
);
$testDs = new MnistDataset(
    "{$base}/t10k-images-idx3-ubyte",
    "{$base}/t10k-labels-idx1-ubyte"
);

echo "Loading {$TRAIN_N} train samples...\n";
$trainX = [];
$trainY = [];
for ($i = 0; $i < $TRAIN_N; $i++) {
    [$x, $y] = $trainDs->get($i);
    $trainX[] = $x->data();
    $trainY[] = (int) $y[0];
}

echo "Loading {$TEST_N} test samples...\n";
$testX = [];
$testY = [];
for ($i = 0; $i < $TEST_N; $i++) {
    [$x, $y] = $testDs->get($i);
    $testX[] = $x->data();
    $testY[] = (int) $y[0];
}

// ====================================================================
// Xavier init
// ====================================================================
mt_srand(42);

$w1 = [];
$scale1 = sqrt(1.0 / IN_DIM);
for ($i = 0; $i < IN_DIM * $HIDDEN; $i++) {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    $w1[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $scale1;
}

$w2 = [];
$scale2 = sqrt(1.0 / $HIDDEN);
for ($i = 0; $i < $HIDDEN * N_CLASSES; $i++) {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    $w2[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $scale2;
}

// ====================================================================
// Allocate GPU buffers (once)
// ====================================================================
$w1Gpu   = $backend->bufferAlloc(IN_DIM * $HIDDEN);   // persistent weights
$xGpu    = $backend->bufferAlloc($BATCH * IN_DIM);    // input batch
$z1Gpu   = $backend->bufferAlloc($BATCH * $HIDDEN);   // hidden activations (pre-ReLU)
$dz1TGpu = $backend->bufferAlloc($HIDDEN * $BATCH);   // transposed hidden grad
$dw1TGpu = $backend->bufferAlloc($HIDDEN * IN_DIM);   // transposed weight grad

$backend->bufferUpload($w1Gpu, $w1);

echo "\n";
echo "Backend: " . $backend->name() . "\n";
echo "Model:   " . IN_DIM . " -> {$HIDDEN} (ReLU) -> " . N_CLASSES . "\n";
echo "Batch:   {$BATCH}\n";
echo "Epochs:  {$EPOCHS}\n";
echo "LR:      {$LR}\n";
echo "Note:    no bias, plain SGD (v1)\n";
echo str_repeat('=', 60) . "\n";

// ====================================================================
// Training
// ====================================================================
$nBatches = (int) ceil($TRAIN_N / $BATCH);
$tStart   = microtime(true);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $epochStart = microtime(true);
    $lossTotal  = 0.0;
    $correct    = 0;
    $seen       = 0;

    $order = range(0, $TRAIN_N - 1);
    shuffle($order);

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * $BATCH;
        $end   = min($start + $BATCH, $TRAIN_N);
        $origN = $end - $start;
        if ($origN === 0) break;

        // ---- Gather batch (pad to BATCH with dummy rows) ----
        $xB = [];
        $yB = [];
        for ($i = $start; $i < $end; $i++) {
            $idx = $order[$i];
            $xB[] = $trainX[$idx];
            $yB[] = $trainY[$idx];
        }
        while (count($xB) < $BATCH) {
            $xB[] = $xB[0];
            $yB[] = -1;   // padding marker
        }

        // ---- Flatten X and upload ----
        $xFlat = [];
        foreach ($xB as $row) {
            foreach ($row as $v) $xFlat[] = $v;
        }
        $backend->bufferUpload($xGpu, $xFlat);

        // ---- Forward matmul on GPU: Z1 = X @ W1 ----
        $backend->matmulDev($xGpu, $w1Gpu, $z1Gpu, $BATCH, IN_DIM, $HIDDEN);
        $z1Flat = $backend->bufferDownload($z1Gpu, $BATCH * $HIDDEN);
        $backend->sync();

        // ---- Unflatten Z1 ----
        $z1 = [];
        for ($i = 0; $i < $BATCH; $i++) {
            $z1[$i] = array_slice($z1Flat, $i * $HIDDEN, $HIDDEN);
        }

        // ---- A1 = ReLU(Z1), keep mask ----
        $a1 = [];
        $mask = [];
        for ($i = 0; $i < $BATCH; $i++) {
            $a1[$i]   = [];
            $mask[$i] = [];
            for ($k = 0; $k < $HIDDEN; $k++) {
                $v = $z1[$i][$k];
                $a1[$i][$k]   = $v > 0 ? $v : 0.0;
                $mask[$i][$k] = $v > 0 ? 1.0 : 0.0;
            }
        }

        // ---- Z2 = A1 @ W2  [BATCH, 10] on CPU ----
        $z2 = [];
        for ($i = 0; $i < $BATCH; $i++) {
            $row = array_fill(0, N_CLASSES, 0.0);
            for ($k = 0; $k < $HIDDEN; $k++) {
                $a = $a1[$i][$k];
                if ($a == 0.0) continue;
                $w2Base = $k * N_CLASSES;
                for ($c = 0; $c < N_CLASSES; $c++) {
                    $row[$c] += $a * $w2[$w2Base + $c];
                }
            }
            $z2[$i] = $row;
        }

        // ---- Softmax + CE loss ----
        $probs = [];
        $batchLoss = 0.0;
        for ($i = 0; $i < $BATCH; $i++) {
            $row = $z2[$i];
            $max = max($row);
            $sum = 0.0;
            for ($c = 0; $c < N_CLASSES; $c++) {
                $row[$c] = exp($row[$c] - $max);
                $sum += $row[$c];
            }
            for ($c = 0; $c < N_CLASSES; $c++) {
                $row[$c] /= $sum;
            }
            $probs[$i] = $row;

            if ($yB[$i] >= 0) {
                $true = $yB[$i];
                $batchLoss += -log(max($row[$true], 1e-12));

                $best = 0; $bv = $row[0];
                for ($c = 1; $c < N_CLASSES; $c++) {
                    if ($row[$c] > $bv) { $bv = $row[$c]; $best = $c; }
                }
                if ($best === $true) $correct++;
                $seen++;
            }
        }
        $batchLoss /= $origN;
        $lossTotal += $batchLoss;

        // ---- dZ2 = (P - one_hot) / origN ----
        // Zero gradient for padding rows.
        $dZ2 = [];
        for ($i = 0; $i < $BATCH; $i++) {
            $row = [];
            $true = $yB[$i];
            if ($true < 0) {
                for ($c = 0; $c < N_CLASSES; $c++) $row[] = 0.0;
            } else {
                for ($c = 0; $c < N_CLASSES; $c++) {
                    $oh = ($c === $true) ? 1.0 : 0.0;
                    $row[] = ($probs[$i][$c] - $oh) / $origN;
                }
            }
            $dZ2[$i] = $row;
        }

        // ---- dW2 = A1^T @ dZ2  [HIDDEN, 10] on CPU ----
        $dW2 = array_fill(0, $HIDDEN * N_CLASSES, 0.0);
        for ($i = 0; $i < $BATCH; $i++) {
            for ($k = 0; $k < $HIDDEN; $k++) {
                $a = $a1[$i][$k];
                if ($a == 0.0) continue;
                $base = $k * N_CLASSES;
                for ($c = 0; $c < N_CLASSES; $c++) {
                    $dW2[$base + $c] += $a * $dZ2[$i][$c];
                }
            }
        }

        // ---- dA1 = dZ2 @ W2^T  [BATCH, HIDDEN] on CPU ----
        $dA1 = [];
        for ($i = 0; $i < $BATCH; $i++) {
            $row = array_fill(0, $HIDDEN, 0.0);
            for ($c = 0; $c < N_CLASSES; $c++) {
                $g = $dZ2[$i][$c];
                if ($g == 0.0) continue;
                for ($k = 0; $k < $HIDDEN; $k++) {
                    $row[$k] += $g * $w2[$k * N_CLASSES + $c];
                }
            }
            $dA1[$i] = $row;
        }

        // ---- dZ1T (transposed and mask-applied)  [HIDDEN, BATCH] ----
        $dZ1T = array_fill(0, $HIDDEN * $BATCH, 0.0);
        for ($i = 0; $i < $BATCH; $i++) {
            for ($k = 0; $k < $HIDDEN; $k++) {
                $dZ1T[$k * $BATCH + $i] = $dA1[$i][$k] * $mask[$i][$k];
            }
        }

        // ---- Backward matmul on GPU: dW1_T = dZ1_T @ X ----
        $backend->bufferUpload($dz1TGpu, $dZ1T);
        $backend->matmulDev($dz1TGpu, $xGpu, $dw1TGpu, $HIDDEN, $BATCH, IN_DIM);
        $dw1TFlat = $backend->bufferDownload($dw1TGpu, $HIDDEN * IN_DIM);
        $backend->sync();

        // ---- Transpose dW1T [HIDDEN, IN_DIM] -> dW1 [IN_DIM, HIDDEN] ----
        $dW1 = array_fill(0, IN_DIM * $HIDDEN, 0.0);
        for ($k = 0; $k < $HIDDEN; $k++) {
            $rowBase = $k * IN_DIM;
            for ($j = 0; $j < IN_DIM; $j++) {
                $dW1[$j * $HIDDEN + $k] = $dw1TFlat[$rowBase + $j];
            }
        }

        // ---- SGD update ----
        $nW1 = count($w1);
        for ($i = 0; $i < $nW1; $i++) {
            $w1[$i] -= $LR * $dW1[$i];
        }
        $nW2 = count($w2);
        for ($i = 0; $i < $nW2; $i++) {
            $w2[$i] -= $LR * $dW2[$i];
        }

        // ---- Re-upload updated W1 to GPU ----
        // (v1: every batch. A v2 with sgd_update_dev on GPU avoids this.)
        $backend->bufferUpload($w1Gpu, $w1);
    }

    $trainAcc = 100.0 * $correct / max($seen, 1);
    $avgLoss  = $lossTotal / $nBatches;
    $elapsed  = microtime(true) - $epochStart;

    printf(
        "epoch %2d/%d   loss=%.4f   train_acc=%.2f%%   %.1fs\n",
        $epoch + 1, $EPOCHS, $avgLoss, $trainAcc, $elapsed
    );
}

$totalTime = microtime(true) - $tStart;

// ====================================================================
// Evaluate
// ====================================================================
echo str_repeat('-', 60) . "\n";
echo "Evaluating {$TEST_N} test samples...\n";

$testCorrect = 0;
for ($start = 0; $start < $TEST_N; $start += $BATCH) {
    $end   = min($start + $BATCH, $TEST_N);
    $nReal = $end - $start;

    $xFlat = [];
    for ($i = $start; $i < $end; $i++) {
        foreach ($testX[$i] as $v) $xFlat[] = $v;
    }
    while (count($xFlat) < $BATCH * IN_DIM) {
        $xFlat[] = 0.0;
    }

    $backend->bufferUpload($xGpu, $xFlat);
    $backend->matmulDev($xGpu, $w1Gpu, $z1Gpu, $BATCH, IN_DIM, $HIDDEN);
    $z1Flat = $backend->bufferDownload($z1Gpu, $BATCH * $HIDDEN);
    $backend->sync();

    for ($i = 0; $i < $nReal; $i++) {
        // ReLU
        $a1 = [];
        $base = $i * $HIDDEN;
        for ($k = 0; $k < $HIDDEN; $k++) {
            $v = $z1Flat[$base + $k];
            $a1[$k] = $v > 0 ? $v : 0.0;
        }
        // Z2
        $z2 = array_fill(0, N_CLASSES, 0.0);
        for ($k = 0; $k < $HIDDEN; $k++) {
            $a = $a1[$k];
            if ($a == 0.0) continue;
            $w2Base = $k * N_CLASSES;
            for ($c = 0; $c < N_CLASSES; $c++) {
                $z2[$c] += $a * $w2[$w2Base + $c];
            }
        }
        // Argmax
        $best = 0; $bv = $z2[0];
        for ($c = 1; $c < N_CLASSES; $c++) {
            if ($z2[$c] > $bv) { $bv = $z2[$c]; $best = $c; }
        }
        if ($best === $testY[$start + $i]) $testCorrect++;
    }
}

$testAcc = 100.0 * $testCorrect / $TEST_N;

echo str_repeat('=', 60) . "\n";
printf("Test accuracy:      %.2f%% (%d / %d)\n", $testAcc, $testCorrect, $TEST_N);
printf("Total training:     %.1f s (%.1f min)\n", $totalTime, $totalTime / 60);
printf("Per epoch:          %.1f s\n", $totalTime / $EPOCHS);

// ====================================================================
// Cleanup
// ====================================================================
$backend->bufferFree($w1Gpu);
$backend->bufferFree($xGpu);
$backend->bufferFree($z1Gpu);
$backend->bufferFree($dz1TGpu);
$backend->bufferFree($dw1TGpu);