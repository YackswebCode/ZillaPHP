<?php

declare(strict_types=1);

/**
 * GPU-native MNIST CNN training.
 *
 * Architecture:
 *   Conv2D(1 → 8,  3×3, pad=1) → ReLU → MaxPool2D(2)
 *   Conv2D(8 → 16, 3×3, pad=1) → ReLU → MaxPool2D(2)
 *   Flatten
 *   Linear(16*7*7 → 10)
 *
 * Every forward and backward operation runs on the GPU via
 * persistent device buffers. Only the raw MNIST images and
 * labels cross PCIe per sample.
 *
 * Simplifications (v1):
 *   - Per-sample processing (batch loop on CPU, ops on device)
 *   - Gradient accumulation on device, SGD update at batch end
 *   - No momentum, no Adam
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
$EPOCHS  = (int)   (getenv('EPOCHS')  ?: 5);
$LR      = (float) (getenv('LR')      ?: 0.01);

// Fixed architecture
const IN_C   = 1;
const IN_H   = 28;
const IN_W   = 28;
const C1     = 8;
const C2     = 16;
const K      = 3;
const PAD    = 1;
const POOL   = 2;
const C1_H   = 28;  const C1_W = 28;      // after conv1 (same due to pad=1)
const P1_H   = 14;  const P1_W = 14;      // after pool1
const C2_H   = 14;  const C2_W = 14;      // after conv2
const P2_H   = 7;   const P2_W = 7;       // after pool2
const FLAT   = C2 * P2_H * P2_W;          // 784
const N_CLS  = 10;

// ====================================================================
// Load MNIST
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
    $trainX[] = $x->data();       // [784] float in [-1, 1]
    $trainY[] = (float) $y[0];
}

echo "Loading {$TEST_N} test samples...\n";
$testX = []; $testY = [];
for ($i = 0; $i < $TEST_N; $i++) {
    [$x, $y] = $testDs->get($i);
    $testX[] = $x->data();
    $testY[] = (int) $y[0];
}

// ====================================================================
// Weight initialization (Xavier)
// ====================================================================
mt_srand(42);

function xavier(int $fanIn, int $fanOut): float {
    return sqrt(2.0 / ($fanIn + $fanOut));
}

function randnArray(int $n, float $scale): array {
    $out = [];
    for ($i = 0; $i < $n; $i++) {
        $u1 = mt_rand() / mt_getrandmax();
        $u2 = mt_rand() / mt_getrandmax();
        $out[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $scale;
    }
    return $out;
}

$w1 = randnArray(C1 * IN_C * K * K, xavier(IN_C * K * K, C1 * K * K));
$b1 = array_fill(0, C1, 0.0);

$w2 = randnArray(C2 * C1 * K * K, xavier(C1 * K * K, C2 * K * K));
$b2 = array_fill(0, C2, 0.0);

$w3 = randnArray(FLAT * N_CLS, xavier(FLAT, N_CLS));
$b3 = array_fill(0, N_CLS, 0.0);

// ====================================================================
// Allocate GPU buffers (once)
// ====================================================================
echo "\nAllocating GPU buffers...\n";

// Weights + biases
$w1Gpu = $backend->bufferAlloc(C1 * IN_C * K * K);
$b1Gpu = $backend->bufferAlloc(C1);
$w2Gpu = $backend->bufferAlloc(C2 * C1 * K * K);
$b2Gpu = $backend->bufferAlloc(C2);
$w3Gpu = $backend->bufferAlloc(FLAT * N_CLS);
$b3Gpu = $backend->bufferAlloc(N_CLS);

// Activations + intermediates
$xGpu    = $backend->bufferAlloc(IN_C * IN_H * IN_W);           // 784
$c1Gpu   = $backend->bufferAlloc(C1 * C1_H * C1_W);             // 6272
$a1Gpu   = $backend->bufferAlloc(C1 * C1_H * C1_W);
$m1Gpu   = $backend->bufferAlloc(C1 * C1_H * C1_W);             // ReLU mask
$p1Gpu   = $backend->bufferAlloc(C1 * P1_H * P1_W);             // 1568
$arg1Gpu = $backend->bufferAlloc(C1 * P1_H * P1_W);             // int-as-float
$c2Gpu   = $backend->bufferAlloc(C2 * C2_H * C2_W);             // 3136
$a2Gpu   = $backend->bufferAlloc(C2 * C2_H * C2_W);
$m2Gpu   = $backend->bufferAlloc(C2 * C2_H * C2_W);
$p2Gpu   = $backend->bufferAlloc(C2 * P2_H * P2_W);             // 784
$arg2Gpu = $backend->bufferAlloc(C2 * P2_H * P2_W);
$z3Gpu   = $backend->bufferAlloc(N_CLS);                        // 10
$dz3Gpu  = $backend->bufferAlloc(N_CLS);
$yGpu    = $backend->bufferAlloc(1);
$lossGpu = $backend->bufferAlloc(1);

// Backward intermediates
$dp2Gpu  = $backend->bufferAlloc(FLAT);
$da2Gpu  = $backend->bufferAlloc(C2 * C2_H * C2_W);
$dc2Gpu  = $backend->bufferAlloc(C2 * C2_H * C2_W);
$dp1Gpu  = $backend->bufferAlloc(C1 * P1_H * P1_W);
$da1Gpu  = $backend->bufferAlloc(C1 * C1_H * C1_W);
$dc1Gpu  = $backend->bufferAlloc(C1 * C1_H * C1_W);
$dxGpu   = $backend->bufferAlloc(IN_C * IN_H * IN_W);

// Gradient accumulators + per-sample temps
$dw1Gpu = $backend->bufferAlloc(C1 * IN_C * K * K);
$db1Gpu = $backend->bufferAlloc(C1);
$dw2Gpu = $backend->bufferAlloc(C2 * C1 * K * K);
$db2Gpu = $backend->bufferAlloc(C2);
$dw3Gpu = $backend->bufferAlloc(FLAT * N_CLS);
$db3Gpu = $backend->bufferAlloc(N_CLS);

$dw1TmpGpu = $backend->bufferAlloc(C1 * IN_C * K * K);
$db1TmpGpu = $backend->bufferAlloc(C1);
$dw2TmpGpu = $backend->bufferAlloc(C2 * C1 * K * K);
$db2TmpGpu = $backend->bufferAlloc(C2);
$dw3TmpGpu = $backend->bufferAlloc(FLAT * N_CLS);
$db3TmpGpu = $backend->bufferAlloc(N_CLS);

// Upload initial weights
$backend->bufferUpload($w1Gpu, $w1);
$backend->bufferUpload($b1Gpu, $b1);
$backend->bufferUpload($w2Gpu, $w2);
$backend->bufferUpload($b2Gpu, $b2);
$backend->bufferUpload($w3Gpu, $w3);
$backend->bufferUpload($b3Gpu, $b3);

// Sizes for zeroing
$SZ_W1 = C1 * IN_C * K * K;
$SZ_W2 = C2 * C1 * K * K;
$SZ_W3 = FLAT * N_CLS;
$SZ_X  = IN_C * IN_H * IN_W;
$SZ_C1 = C1 * C1_H * C1_W;
$SZ_P1 = C1 * P1_H * P1_W;
$SZ_C2 = C2 * C2_H * C2_W;
$SZ_P2 = C2 * P2_H * P2_W;

echo "Backend: " . $backend->name() . "\n";
echo "Architecture: Conv(1→8,3×3,p=1) → ReLU → Pool(2) → Conv(8→16,3×3,p=1) → ReLU → Pool(2) → Linear(784→10)\n";
echo "Train samples: {$TRAIN_N}   Test samples: {$TEST_N}\n";
echo "Batch: {$BATCH}   Epochs: {$EPOCHS}   LR: {$LR}\n";
echo str_repeat('=', 60) . "\n";

// ====================================================================
// Training loop
// ====================================================================
$nBatches = (int) ceil($TRAIN_N / $BATCH);
$tStart   = microtime(true);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $epochStart = microtime(true);
    $epochLoss  = 0.0;

    $order = range(0, $TRAIN_N - 1);
    shuffle($order);

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * $BATCH;
        $end   = min($start + $BATCH, $TRAIN_N);
        $N     = $end - $start;
        if ($N === 0) break;

        // ---- Zero gradient accumulators ----
        $backend->zeroDev($dw1Gpu, $SZ_W1);
        $backend->zeroDev($db1Gpu, C1);
        $backend->zeroDev($dw2Gpu, $SZ_W2);
        $backend->zeroDev($db2Gpu, C2);
        $backend->zeroDev($dw3Gpu, $SZ_W3);
        $backend->zeroDev($db3Gpu, N_CLS);
        $backend->zeroDev($lossGpu, 1);

        // ---- Per-sample forward + backward ----
        for ($n = 0; $n < $N; $n++) {
            $idx = $order[$start + $n];

            // Upload input
            $backend->bufferUpload($xGpu, $trainX[$idx]);
            $backend->bufferUpload($yGpu, [$trainY[$idx]]);

            // ---- Forward ----
            $backend->conv2dFwdDev(
                $xGpu, $w1Gpu, $b1Gpu, $c1Gpu,
                IN_C, IN_H, IN_W,
                C1, C1_H, C1_W,
                K, K, 1, PAD
            );
            $backend->reluFwdDev($c1Gpu, $a1Gpu, $m1Gpu, $SZ_C1);
            $backend->maxPool2dFwdDev(
                $a1Gpu, $p1Gpu, $arg1Gpu,
                C1, C1_H, C1_W,
                P1_H, P1_W,
                POOL, POOL, POOL
            );
            $backend->conv2dFwdDev(
                $p1Gpu, $w2Gpu, $b2Gpu, $c2Gpu,
                C1, P1_H, P1_W,
                C2, C2_H, C2_W,
                K, K, 1, PAD
            );
            $backend->reluFwdDev($c2Gpu, $a2Gpu, $m2Gpu, $SZ_C2);
            $backend->maxPool2dFwdDev(
                $a2Gpu, $p2Gpu, $arg2Gpu,
                C2, C2_H, C2_W,
                P2_H, P2_W,
                POOL, POOL, POOL
            );
            // p2 is [C2, 7, 7] = [784] in memory → treat as [1, 784]
            $backend->matmulDev($p2Gpu, $w3Gpu, $z3Gpu, 1, FLAT, N_CLS);
            $backend->addBiasDev($z3Gpu, $b3Gpu, 1, N_CLS);

            // Loss + dZ3
            $backend->softmaxCeDev($z3Gpu, $yGpu, $dz3Gpu, $lossGpu, 1, N_CLS);

            // ---- Backward ----
            // dW3 = p2ᵀ @ dZ3   (p2 as [K=1, M=784], dZ3 as [K=1, N=10])
            $backend->matmulTnDev($p2Gpu, $dz3Gpu, $dw3TmpGpu, FLAT, 1, N_CLS);
            $backend->addDev($dw3Gpu, $dw3TmpGpu, $dw3Gpu, $SZ_W3);
            $backend->biasGradDev($dz3Gpu, $db3TmpGpu, 1, N_CLS);
            $backend->addDev($db3Gpu, $db3TmpGpu, $db3Gpu, N_CLS);

            // dp2 = dZ3 @ w3ᵀ
            $backend->matmulNtDev($dz3Gpu, $w3Gpu, $dp2Gpu, 1, N_CLS, FLAT);

            // MaxPool2D backward (zero gradInput first)
            $backend->zeroDev($da2Gpu, $SZ_C2);
            $backend->maxPool2dBwdDev($dp2Gpu, $arg2Gpu, $da2Gpu, C2, P2_H, P2_W);

            // ReLU backward
            $backend->reluBwdDev($da2Gpu, $m2Gpu, $dc2Gpu, $SZ_C2);

            // Conv2D backward (input=p1, weight=w2, gradOut=dc2)
            $backend->zeroDev($dp1Gpu, $SZ_P1);
            $backend->conv2dBwdDev(
                $p1Gpu, $w2Gpu, $dc2Gpu,
                $dp1Gpu, $dw2TmpGpu, $db2TmpGpu,
                C1, P1_H, P1_W,
                C2, C2_H, C2_W,
                K, K, 1, PAD
            );
            $backend->addDev($dw2Gpu, $dw2TmpGpu, $dw2Gpu, $SZ_W2);
            $backend->addDev($db2Gpu, $db2TmpGpu, $db2Gpu, C2);

            // MaxPool2D backward
            $backend->zeroDev($da1Gpu, $SZ_C1);
            $backend->maxPool2dBwdDev($dp1Gpu, $arg1Gpu, $da1Gpu, C1, P1_H, P1_W);

            // ReLU backward
            $backend->reluBwdDev($da1Gpu, $m1Gpu, $dc1Gpu, $SZ_C1);

            // Conv2D backward (input=x, weight=w1, gradOut=dc1)
            $backend->zeroDev($dxGpu, $SZ_X);
            $backend->conv2dBwdDev(
                $xGpu, $w1Gpu, $dc1Gpu,
                $dxGpu, $dw1TmpGpu, $db1TmpGpu,
                IN_C, IN_H, IN_W,
                C1, C1_H, C1_W,
                K, K, 1, PAD
            );
            $backend->addDev($dw1Gpu, $dw1TmpGpu, $dw1Gpu, $SZ_W1);
            $backend->addDev($db1Gpu, $db1TmpGpu, $db1Gpu, C1);
        }

        // ---- SGD update after batch ----
        $lrEff = $LR / $N;   // average the per-sample gradients
        $backend->sgdUpdateDev($w1Gpu, $dw1Gpu, $lrEff, $SZ_W1);
        $backend->sgdUpdateDev($b1Gpu, $db1Gpu, $lrEff, C1);
        $backend->sgdUpdateDev($w2Gpu, $dw2Gpu, $lrEff, $SZ_W2);
        $backend->sgdUpdateDev($b2Gpu, $db2Gpu, $lrEff, C2);
        $backend->sgdUpdateDev($w3Gpu, $dw3Gpu, $lrEff, $SZ_W3);
        $backend->sgdUpdateDev($b3Gpu, $db3Gpu, $lrEff, N_CLS);
    }

    $backend->sync();
    $epochSec = microtime(true) - $epochStart;
    printf("epoch %2d/%d   %.1fs\n", $epoch + 1, $EPOCHS, $epochSec);
}

$trainSec = microtime(true) - $tStart;

// ====================================================================
// Evaluate — download weights once, run on CPU
// ====================================================================
echo str_repeat('-', 60) . "\n";
echo "Downloading trained weights...\n";
$backend->sync();
$w1T = $backend->bufferDownload($w1Gpu, $SZ_W1);
$b1T = $backend->bufferDownload($b1Gpu, C1);
$w2T = $backend->bufferDownload($w2Gpu, $SZ_W2);
$b2T = $backend->bufferDownload($b2Gpu, C2);
$w3T = $backend->bufferDownload($w3Gpu, $SZ_W3);
$b3T = $backend->bufferDownload($b3Gpu, N_CLS);

echo "Evaluating {$TEST_N} test samples (CPU)...\n";
$correct = 0;

for ($i = 0; $i < $TEST_N; $i++) {
    $img = $testX[$i];  // [784] in [0, 1] from MnistDataset

    // Reshape to [1, 28, 28] — MnistDataset returns [0,1] range, subtract mean
    // Actually keep in [0, 1] since we trained on [0, 1]

    // ---- Conv1 forward (CPU PHP for eval) ----
    // This is slow but fine for a one-time eval pass
    $c1 = array_fill(0, C1 * C1_H * C1_W, 0.0);
    for ($oc = 0; $oc < C1; $oc++) {
        for ($oh = 0; $oh < C1_H; $oh++) {
            for ($ow = 0; $ow < C1_W; $ow++) {
                $sum = $b1T[$oc];
                for ($kh = 0; $kh < K; $kh++) {
                    $ih = $oh - PAD + $kh;
                    if ($ih < 0 || $ih >= IN_H) continue;
                    for ($kw = 0; $kw < K; $kw++) {
                        $iw = $ow - PAD + $kw;
                        if ($iw < 0 || $iw >= IN_W) continue;
                        $sum += $img[$ih * IN_W + $iw] * $w1T[($oc * 9) + $kh * 3 + $kw];
                    }
                }
                $c1[$oc * C1_H * C1_W + $oh * C1_W + $ow] = $sum > 0 ? $sum : 0.0;
            }
        }
    }

    // MaxPool1
    $p1 = [];
    for ($c = 0; $c < C1; $c++) {
        for ($oh = 0; $oh < P1_H; $oh++) {
            for ($ow = 0; $ow < P1_W; $ow++) {
                $best = -1e30;
                for ($kh = 0; $kh < POOL; $kh++) {
                    for ($kw = 0; $kw < POOL; $kw++) {
                        $ih = $oh * POOL + $kh;
                        $iw = $ow * POOL + $kw;
                        $v = $c1[$c * C1_H * C1_W + $ih * C1_W + $iw];
                        if ($v > $best) $best = $v;
                    }
                }
                $p1[] = $best;
            }
        }
    }

    // Conv2
    $c2 = array_fill(0, C2 * C2_H * C2_W, 0.0);
    for ($oc = 0; $oc < C2; $oc++) {
        for ($oh = 0; $oh < C2_H; $oh++) {
            for ($ow = 0; $ow < C2_W; $ow++) {
                $sum = $b2T[$oc];
                for ($ic = 0; $ic < C1; $ic++) {
                    $pBase = $ic * P1_H * P1_W;
                    $wBase = ($oc * C1 + $ic) * 9;
                    for ($kh = 0; $kh < K; $kh++) {
                        $ih = $oh - PAD + $kh;
                        if ($ih < 0 || $ih >= P1_H) continue;
                        for ($kw = 0; $kw < K; $kw++) {
                            $iw = $ow - PAD + $kw;
                            if ($iw < 0 || $iw >= P1_W) continue;
                            $sum += $p1[$pBase + $ih * P1_W + $iw] * $w2T[$wBase + $kh * 3 + $kw];
                        }
                    }
                }
                $c2[$oc * C2_H * C2_W + $oh * C2_W + $ow] = $sum > 0 ? $sum : 0.0;
            }
        }
    }

    // MaxPool2
    $p2 = [];
    for ($c = 0; $c < C2; $c++) {
        for ($oh = 0; $oh < P2_H; $oh++) {
            for ($ow = 0; $ow < P2_W; $ow++) {
                $best = -1e30;
                for ($kh = 0; $kh < POOL; $kh++) {
                    for ($kw = 0; $kw < POOL; $kw++) {
                        $ih = $oh * POOL + $kh;
                        $iw = $ow * POOL + $kw;
                        $v = $c2[$c * C2_H * C2_W + $ih * C2_W + $iw];
                        if ($v > $best) $best = $v;
                    }
                }
                $p2[] = $best;
            }
        }
    }

    // Linear
    $logits = $b3T;
    for ($j = 0; $j < FLAT; $j++) {
        $v = $p2[$j];
        if ($v == 0.0) continue;
        $base = $j * N_CLS;
        for ($k = 0; $k < N_CLS; $k++) {
            $logits[$k] += $v * $w3T[$base + $k];
        }
    }

    $best = 0; $bv = $logits[0];
    for ($k = 1; $k < N_CLS; $k++) if ($logits[$k] > $bv) { $bv = $logits[$k]; $best = $k; }
    if ($best === $testY[$i]) $correct++;
}

$acc = 100.0 * $correct / $TEST_N;

echo str_repeat('=', 60) . "\n";
printf("Test accuracy:   %.2f%% (%d / %d)\n", $acc, $correct, $TEST_N);
printf("Total training:  %.1f s (%.1f min)\n", $trainSec, $trainSec / 60);
printf("Per epoch:       %.1f s\n", $trainSec / $EPOCHS);

// ====================================================================
// Cleanup
// ====================================================================
$allBuffers = [
    $w1Gpu, $b1Gpu, $w2Gpu, $b2Gpu, $w3Gpu, $b3Gpu,
    $xGpu, $c1Gpu, $a1Gpu, $m1Gpu, $p1Gpu, $arg1Gpu,
    $c2Gpu, $a2Gpu, $m2Gpu, $p2Gpu, $arg2Gpu,
    $z3Gpu, $dz3Gpu, $yGpu, $lossGpu,
    $dp2Gpu, $da2Gpu, $dc2Gpu, $dp1Gpu, $da1Gpu, $dc1Gpu, $dxGpu,
    $dw1Gpu, $db1Gpu, $dw2Gpu, $db2Gpu, $dw3Gpu, $db3Gpu,
    $dw1TmpGpu, $db1TmpGpu, $dw2TmpGpu, $db2TmpGpu, $dw3TmpGpu, $db3TmpGpu,
];
foreach ($allBuffers as $id) $backend->bufferFree($id);