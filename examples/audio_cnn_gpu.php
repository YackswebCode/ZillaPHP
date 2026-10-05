<?php

declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\SpeechCommandsDataset;
use ZillaPHP\Hardware\CUDA\CudaBackend;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;

Application::boot(cuda: true);

$backend = \ZillaPHP\Tensor\Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required.\n");
    exit(1);
}

$MAX_PER_CLASS = (int)   (getenv('MAX_PER_CLASS') ?: 500);
$BATCH         = (int)   (getenv('BATCH')         ?: 32);
$EPOCHS        = (int)   (getenv('EPOCHS')        ?: 10);
$LR            = (float) (getenv('LR')            ?: 0.01);

// ---- Shape config ----
const IN_C = 1;
const IN_H = 64;   // 64 mel bands
const IN_W = 101;  // 101 frames (1 second at 16 kHz, 160-hop)

const C1 = 8;
const K  = 3;
const PAD = 1;
const C2 = 16;
const POOL = 2;

// After Conv1 (pad=1):    [64, 101]
// After Pool1 (2):        [32, 50]
// After Conv2 (pad=1):    [32, 50]
// After Pool2 (2):        [16, 25]
const C1_H = 64;  const C1_W = 101;
const P1_H = 32;  const P1_W = 50;
const C2_H = 32;  const C2_W = 50;
const P2_H = 16;  const P2_W = 25;
const FLAT = C2 * P2_H * P2_W;   // 16 * 16 * 25 = 6400
const N_CLS = 10;

// ---- Load data ----
$root = __DIR__ . '/../data/speech_commands';
if (!is_dir($root)) {
    fwrite(STDERR, "Speech Commands not found. Run:\n");
    fwrite(STDERR, "  php scripts/download_speech_commands.php --subset\n");
    exit(1);
}

echo "Building dataset (max {$MAX_PER_CLASS} per class)...\n";

$cpuBackend = new NativeCpuBackend();

$trainDs = new SpeechCommandsDataset($root, 'train', $MAX_PER_CLASS, $cpuBackend);
$testDs  = new SpeechCommandsDataset($root, 'test',  100, $cpuBackend);

echo "Classes: " . implode(', ', $trainDs->classes()) . "\n";
echo "Train size: " . $trainDs->size() . "\n";
echo "Test size:  " . $testDs->size() . "\n\n";

// Sanity: prepack tensors into flat arrays for upload speed
echo "Preparing training data...\n";
$trainX = []; $trainY = [];
$t0 = microtime(true);
for ($i = 0; $i < $trainDs->size(); $i++) {
    [$t, $y] = $trainDs->get($i);
    $trainX[] = $t->data();
    $trainY[] = (float) $y;
    if (($i + 1) % 500 === 0) {
        printf("  %d / %d  (%.1fs)\n", $i + 1, $trainDs->size(), microtime(true) - $t0);
    }
}
echo "  done (" . round(microtime(true) - $t0, 1) . "s)\n\n";

$testX = []; $testY = [];
for ($i = 0; $i < $testDs->size(); $i++) {
    [$t, $y] = $testDs->get($i);
    $testX[] = $t->data();
    $testY[] = $y;
}
echo "Prepared " . count($trainX) . " train + " . count($testX) . " test samples.\n\n";

// ---- Init weights (Xavier) ----
mt_srand(42);
function xavier(int $fIn, int $fOut): float { return sqrt(2.0 / ($fIn + $fOut)); }
function randnArr(int $n, float $s): array {
    $o = [];
    for ($i = 0; $i < $n; $i++) {
        $u1 = mt_rand() / mt_getrandmax();
        $u2 = mt_rand() / mt_getrandmax();
        $o[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $s;
    }
    return $o;
}

$SZ_W1 = C1 * IN_C * K * K;
$SZ_W2 = C2 * C1 * K * K;
$SZ_W3 = FLAT * N_CLS;

$w1 = randnArr($SZ_W1, xavier(IN_C * K * K, C1 * K * K));
$b1 = array_fill(0, C1, 0.0);
$w2 = randnArr($SZ_W2, xavier(C1 * K * K, C2 * K * K));
$b2 = array_fill(0, C2, 0.0);
$w3 = randnArr($SZ_W3, xavier(FLAT, N_CLS));
$b3 = array_fill(0, N_CLS, 0.0);

// ---- Allocate GPU buffers ----
echo "Allocating GPU buffers...\n";
$SZ_X  = $BATCH * IN_C * IN_H * IN_W;
$SZ_C1 = $BATCH * C1 * C1_H * C1_W;
$SZ_P1 = $BATCH * C1 * P1_H * P1_W;
$SZ_C2 = $BATCH * C2 * C2_H * C2_W;
$SZ_P2 = $BATCH * C2 * P2_H * P2_W;

$w1Gpu = $backend->bufferAlloc($SZ_W1);
$b1Gpu = $backend->bufferAlloc(C1);
$w2Gpu = $backend->bufferAlloc($SZ_W2);
$b2Gpu = $backend->bufferAlloc(C2);
$w3Gpu = $backend->bufferAlloc($SZ_W3);
$b3Gpu = $backend->bufferAlloc(N_CLS);

$xGpu  = $backend->bufferAlloc($SZ_X);
$c1Gpu = $backend->bufferAlloc($SZ_C1);
$a1Gpu = $backend->bufferAlloc($SZ_C1);
$m1Gpu = $backend->bufferAlloc($SZ_C1);
$p1Gpu = $backend->bufferAlloc($SZ_P1);
$arg1Gpu = $backend->bufferAlloc($SZ_P1);
$c2Gpu = $backend->bufferAlloc($SZ_C2);
$a2Gpu = $backend->bufferAlloc($SZ_C2);
$m2Gpu = $backend->bufferAlloc($SZ_C2);
$p2Gpu = $backend->bufferAlloc($SZ_P2);
$arg2Gpu = $backend->bufferAlloc($SZ_P2);
$z3Gpu = $backend->bufferAlloc($BATCH * N_CLS);
$dz3Gpu = $backend->bufferAlloc($BATCH * N_CLS);
$yGpu = $backend->bufferAlloc($BATCH);
$lossGpu = $backend->bufferAlloc(1);

$dp2Gpu = $backend->bufferAlloc($SZ_P2);
$da2Gpu = $backend->bufferAlloc($SZ_C2);
$dc2Gpu = $backend->bufferAlloc($SZ_C2);
$dp1Gpu = $backend->bufferAlloc($SZ_P1);
$da1Gpu = $backend->bufferAlloc($SZ_C1);
$dc1Gpu = $backend->bufferAlloc($SZ_C1);
$dxGpu  = $backend->bufferAlloc($SZ_X);

$dw1Gpu = $backend->bufferAlloc($SZ_W1);
$db1Gpu = $backend->bufferAlloc(C1);
$dw2Gpu = $backend->bufferAlloc($SZ_W2);
$db2Gpu = $backend->bufferAlloc(C2);
$dw3Gpu = $backend->bufferAlloc($SZ_W3);
$db3Gpu = $backend->bufferAlloc(N_CLS);

$backend->bufferUpload($w1Gpu, $w1);
$backend->bufferUpload($b1Gpu, $b1);
$backend->bufferUpload($w2Gpu, $w2);
$backend->bufferUpload($b2Gpu, $b2);
$backend->bufferUpload($w3Gpu, $w3);
$backend->bufferUpload($b3Gpu, $b3);

echo "Backend: " . $backend->name() . "\n";
echo "Architecture: Conv(1→8) → ReLU → Pool(2) → Conv(8→16) → ReLU → Pool(2) → FC(6400→10)\n";
echo "Batch: {$BATCH}   Epochs: {$EPOCHS}   LR: {$LR}\n";
echo str_repeat('=', 60) . "\n";

$N = count($trainX);
$nBatches = (int) ceil($N / $BATCH);
$tStart = microtime(true);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $epochStart = microtime(true);
    $order = range(0, $N - 1);
    shuffle($order);

    $epochLoss = 0.0;

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * $BATCH;
        $end   = min($start + $BATCH, $N);
        $nReal = $end - $start;
        if ($nReal === 0) break;

        // Assemble batch (pad with last sample)
        $xFlat = [];
        $yFlat = [];
        for ($i = $start; $i < $end; $i++) {
            $idx = $order[$i];
            foreach ($trainX[$idx] as $v) $xFlat[] = $v;
            $yFlat[] = $trainY[$idx];
        }
        while (count($yFlat) < $BATCH) {
            $idx = $order[$start];
            foreach ($trainX[$idx] as $v) $xFlat[] = $v;
            $yFlat[] = $trainY[$idx];
        }

        $backend->bufferUpload($xGpu, $xFlat);
        $backend->bufferUpload($yGpu, $yFlat);
        $backend->zeroDev($lossGpu, 1);

        // ---- Forward ----
        $backend->conv2dBatchedFwdDev($xGpu, $w1Gpu, $b1Gpu, $c1Gpu,
            $BATCH, IN_C, IN_H, IN_W, C1, C1_H, C1_W, K, K, 1, PAD);
        $backend->reluFwdDev($c1Gpu, $a1Gpu, $m1Gpu, $SZ_C1);
        $backend->maxPool2dBatchedFwdDev($a1Gpu, $p1Gpu, $arg1Gpu,
            $BATCH, C1, C1_H, C1_W, P1_H, P1_W, POOL, POOL, POOL);
        $backend->conv2dBatchedFwdDev($p1Gpu, $w2Gpu, $b2Gpu, $c2Gpu,
            $BATCH, C1, P1_H, P1_W, C2, C2_H, C2_W, K, K, 1, PAD);
        $backend->reluFwdDev($c2Gpu, $a2Gpu, $m2Gpu, $SZ_C2);
        $backend->maxPool2dBatchedFwdDev($a2Gpu, $p2Gpu, $arg2Gpu,
            $BATCH, C2, C2_H, C2_W, P2_H, P2_W, POOL, POOL, POOL);

        $backend->matmulDev($p2Gpu, $w3Gpu, $z3Gpu, $BATCH, FLAT, N_CLS);
        $backend->addBiasDev($z3Gpu, $b3Gpu, $BATCH, N_CLS);
        $backend->softmaxCeDev($z3Gpu, $yGpu, $dz3Gpu, $lossGpu, $BATCH, N_CLS);

        // ---- Backward ----
        $backend->zeroDev($dw3Gpu, $SZ_W3);
        $backend->matmulTnDev($p2Gpu, $dz3Gpu, $dw3Gpu, FLAT, $BATCH, N_CLS);
        $backend->zeroDev($db3Gpu, N_CLS);
        $backend->biasGradDev($dz3Gpu, $db3Gpu, $BATCH, N_CLS);
        $backend->matmulNtDev($dz3Gpu, $w3Gpu, $dp2Gpu, $BATCH, N_CLS, FLAT);
        $backend->zeroDev($da2Gpu, $SZ_C2);
        $backend->maxPool2dBatchedBwdDev($dp2Gpu, $arg2Gpu, $da2Gpu, $BATCH, C2, P2_H, P2_W);
        $backend->reluBwdDev($da2Gpu, $m2Gpu, $dc2Gpu, $SZ_C2);

        $backend->zeroDev($dp1Gpu, $SZ_P1);
        $backend->zeroDev($dw2Gpu, $SZ_W2);
        $backend->zeroDev($db2Gpu, C2);
        $backend->conv2dBatchedBwdDev($p1Gpu, $w2Gpu, $dc2Gpu,
            $dp1Gpu, $dw2Gpu, $db2Gpu,
            $BATCH, C1, P1_H, P1_W, C2, C2_H, C2_W, K, K, 1, PAD);

        $backend->zeroDev($da1Gpu, $SZ_C1);
        $backend->maxPool2dBatchedBwdDev($dp1Gpu, $arg1Gpu, $da1Gpu, $BATCH, C1, P1_H, P1_W);
        $backend->reluBwdDev($da1Gpu, $m1Gpu, $dc1Gpu, $SZ_C1);

        $backend->zeroDev($dxGpu, $SZ_X);
        $backend->zeroDev($dw1Gpu, $SZ_W1);
        $backend->zeroDev($db1Gpu, C1);
        $backend->conv2dBatchedBwdDev($xGpu, $w1Gpu, $dc1Gpu,
            $dxGpu, $dw1Gpu, $db1Gpu,
            $BATCH, IN_C, IN_H, IN_W, C1, C1_H, C1_W, K, K, 1, PAD);

        // ---- SGD ----
        $backend->sgdUpdateDev($w1Gpu, $dw1Gpu, $LR, $SZ_W1);
        $backend->sgdUpdateDev($b1Gpu, $db1Gpu, $LR, C1);
        $backend->sgdUpdateDev($w2Gpu, $dw2Gpu, $LR, $SZ_W2);
        $backend->sgdUpdateDev($b2Gpu, $db2Gpu, $LR, C2);
        $backend->sgdUpdateDev($w3Gpu, $dw3Gpu, $LR, $SZ_W3);
        $backend->sgdUpdateDev($b3Gpu, $db3Gpu, $LR, N_CLS);

        $epochLoss += (float) $backend->bufferDownload($lossGpu, 1)[0];
    }

    $epochSec = microtime(true) - $epochStart;
    printf("epoch %2d/%d   loss=%.4f   %.1fs\n",
        $epoch + 1, $EPOCHS, $epochLoss / $nBatches, $epochSec);
}

$trainSec = microtime(true) - $tStart;

// ---- Eval ----
echo str_repeat('-', 60) . "\n";
echo "Evaluating " . count($testX) . " test samples...\n";

// Download weights for CPU eval
$backend->sync();
$w1T = $backend->bufferDownload($w1Gpu, $SZ_W1);
$b1T = $backend->bufferDownload($b1Gpu, C1);
$w2T = $backend->bufferDownload($w2Gpu, $SZ_W2);
$b2T = $backend->bufferDownload($b2Gpu, C2);
$w3T = $backend->bufferDownload($w3Gpu, $SZ_W3);
$b3T = $backend->bufferDownload($b3Gpu, N_CLS);

$correct = 0;
foreach ($testX as $i => $img) {
    // Conv1
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
    // Pool1
    $p1 = [];
    for ($c = 0; $c < C1; $c++) {
        for ($oh = 0; $oh < P1_H; $oh++) {
            for ($ow = 0; $ow < P1_W; $ow++) {
                $best = -1e30;
                for ($kh = 0; $kh < POOL; $kh++)
                    for ($kw = 0; $kw < POOL; $kw++) {
                        $v = $c1[$c * C1_H * C1_W + ($oh*POOL+$kh) * C1_W + ($ow*POOL+$kw)];
                        if ($v > $best) $best = $v;
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
                    $wBase = ($oc * C1 + $ic) * 9;
                    for ($kh = 0; $kh < K; $kh++) {
                        $ih = $oh - PAD + $kh;
                        if ($ih < 0 || $ih >= P1_H) continue;
                        for ($kw = 0; $kw < K; $kw++) {
                            $iw = $ow - PAD + $kw;
                            if ($iw < 0 || $iw >= P1_W) continue;
                            $sum += $p1[$ic * P1_H * P1_W + $ih * P1_W + $iw] * $w2T[$wBase + $kh * 3 + $kw];
                        }
                    }
                }
                $c2[$oc * C2_H * C2_W + $oh * C2_W + $ow] = $sum > 0 ? $sum : 0.0;
            }
        }
    }
    // Pool2
    $p2 = [];
    for ($c = 0; $c < C2; $c++) {
        for ($oh = 0; $oh < P2_H; $oh++) {
            for ($ow = 0; $ow < P2_W; $ow++) {
                $best = -1e30;
                for ($kh = 0; $kh < POOL; $kh++)
                    for ($kw = 0; $kw < POOL; $kw++) {
                        $v = $c2[$c * C2_H * C2_W + ($oh*POOL+$kh) * C2_W + ($ow*POOL+$kw)];
                        if ($v > $best) $best = $v;
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
        for ($k = 0; $k < N_CLS; $k++) $logits[$k] += $v * $w3T[$base + $k];
    }
    $best = 0;
    for ($k = 1; $k < N_CLS; $k++) if ($logits[$k] > $logits[$best]) $best = $k;
    if ($best === $testY[$i]) $correct++;
}

$acc = 100.0 * $correct / count($testX);

echo str_repeat('=', 60) . "\n";
printf("Test accuracy:   %.2f%% (%d / %d)\n", $acc, $correct, count($testX));
printf("Total training:  %.1f s (%.1f min)\n", $trainSec, $trainSec / 60);
printf("Per epoch:       %.1f s\n", $trainSec / $EPOCHS);