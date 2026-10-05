<?php

declare(strict_types=1);

/**
 * Synthetic audio classification — no download required.
 *
 * Generates 10 classes of 1-second audio clips at 16 kHz:
 *   - 8 single-frequency sines (200, 400, 600, 800, 1000, 1500, 2000, 3000 Hz)
 *   - 1 sweep (200→4000 Hz)
 *   - 1 noise burst
 *
 * Each clip is converted to log-mel [1, 64, 101] and fed to the same
 * batched CNN used for MNIST. This proves the audio pipeline works
 * end-to-end without needing the Speech Commands download.
 */

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;
use ZillaPHP\Hardware\CUDA\CudaBackend;

Application::boot(cuda: true);

$backend = \ZillaPHP\Tensor\Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required.\n");
    exit(1);
}

$SAMPLES_PER_CLASS = (int)   (getenv('SAMPLES_PER_CLASS') ?: 100);
$BATCH             = (int)   (getenv('BATCH')             ?: 32);
$EPOCHS            = (int)   (getenv('EPOCHS')            ?: 10);
$LR                = (float) (getenv('LR')                ?: 0.01);

// ---- Audio parameters ----
const SR       = 16000;
const N_SAMP   = 16000;
const FFT      = 512;
const HOP      = 160;
const N_MELS   = 64;
const N_FRAMES = 101;
const N_CLASS  = 10;

// ---- CNN shape (matching the audio CNN example) ----
const IN_C = 1; const IN_H = N_MELS; const IN_W = N_FRAMES;
const C1 = 8;   const K = 3; const PAD = 1;
const C2 = 16;  const POOL = 2;
const C1_H = 64; const C1_W = 101;
const P1_H = 32; const P1_W = 50;
const C2_H = 32; const C2_W = 50;
const P2_H = 16; const P2_W = 25;
const FLAT = C2 * P2_H * P2_W;   // 6400

// ====================================================================
// 1. Generate synthetic audio → log-mel features
// ====================================================================
echo "Generating synthetic audio dataset...\n";
echo "  classes: 10 (8 sines, 1 sweep, 1 noise)\n";
echo "  samples per class: {$SAMPLES_PER_CLASS}\n";

$cpuBackend = new NativeCpuBackend();
$window = $cpuBackend->hannWindow(FFT);

$melBank = new \ZillaPHP\Audio\MelFilterbank(SR, FFT, N_MELS);
$melFlat = $melBank->flattened();

/**
 * Convert a waveform (float[]) to a flat [64*101] log-mel.
 */
function toLogMel(array $samples, array $window, array $melFlat, NativeCpuBackend $cpu): array
{
    // Pad/trim to exactly 16000 samples
    $n = count($samples);
    if ($n < N_SAMP) $samples = array_merge($samples, array_fill(0, N_SAMP - $n, 0.0));
    elseif ($n > N_SAMP) $samples = array_slice($samples, 0, N_SAMP);

    [$mag, $nBins, $nFrames] = $cpu->stftMagnitude($samples, $window, FFT, HOP);

    $out = array_fill(0, N_MELS * N_FRAMES, -10.0);
    for ($m = 0; $m < N_MELS; $m++) {
        for ($f = 0; $f < $nFrames; $f++) {
            $sum = 0.0;
            for ($b = 0; $b < $nBins; $b++) {
                $w = $melFlat[$m * 257 + $b];
                if ($w == 0.0) continue;
                $sum += $w * $mag[$b * $nFrames + $f];
            }
            $out[$m * N_FRAMES + $f] = log($sum + 1e-6);
        }
    }
    return $out;
}

/** Generate one training sample for class $c with small random variation. */
function generateSample(int $c, callable $randn): array
{
    $samples = [];
    $freqs = [200, 400, 600, 800, 1000, 1500, 2000, 3000];
    $amp = 0.5;

    if ($c < 8) {
        // Pure sine with tiny frequency jitter
        $f = $freqs[$c] * (1.0 + $randn() * 0.02);
        for ($i = 0; $i < N_SAMP; $i++) {
            $samples[] = $amp * sin(2 * M_PI * $f * $i / SR);
        }
    } elseif ($c === 8) {
        // Linear sweep 200 → 4000 Hz
        $f0 = 200; $f1 = 4000;
        for ($i = 0; $i < N_SAMP; $i++) {
            $t = $i / SR;
            $instF = $f0 + ($f1 - $f0) * ($t / 1.0);
            $samples[] = $amp * sin(2 * M_PI * $instF * $t);
        }
    } else {
        // Noise
        for ($i = 0; $i < N_SAMP; $i++) {
            $samples[] = 0.3 * $randn();
        }
    }

    // Small amplitude noise (augmentation)
    foreach ($samples as $i => $v) {
        $samples[$i] = $v + 0.05 * $randn();
    }

    return $samples;
}

/** Box-Muller random normal. */
$randn = function (): float {
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    return sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2);
};

mt_srand(42);

// Generate train set
$trainX = []; $trainY = [];
$t0 = microtime(true);
$total = $SAMPLES_PER_CLASS * N_CLASS;
for ($c = 0; $c < N_CLASS; $c++) {
    for ($i = 0; $i < $SAMPLES_PER_CLASS; $i++) {
        $samples = generateSample($c, $randn);
        $trainX[] = toLogMel($samples, $window, $melFlat, $cpuBackend);
        $trainY[] = (float) $c;
    }
    printf("  class %d: %d samples (%.1fs)\n", $c, $SAMPLES_PER_CLASS, microtime(true) - $t0);
}
echo "  total: " . count($trainX) . " train samples (" . round(microtime(true) - $t0, 1) . "s)\n\n";

// Generate test set (different random seed effectively)
$testX = []; $testY = [];
for ($c = 0; $c < N_CLASS; $c++) {
    for ($i = 0; $i < 20; $i++) {
        $samples = generateSample($c, $randn);
        $testX[] = toLogMel($samples, $window, $melFlat, $cpuBackend);
        $testY[] = $c;
    }
}
echo "Generated " . count($testX) . " test samples.\n\n";

// ====================================================================
// 2. Init weights
// ====================================================================
function xavier(int $fIn, int $fOut): float { return sqrt(2.0 / ($fIn + $fOut)); }
function randnArr(int $n, float $s, callable $randn): array {
    $o = [];
    for ($i = 0; $i < $n; $i++) $o[] = $randn() * $s;
    return $o;
}

$SZ_W1 = C1 * IN_C * K * K;
$SZ_W2 = C2 * C1 * K * K;
$SZ_W3 = FLAT * N_CLASS;

$w1 = randnArr($SZ_W1, xavier(IN_C * K * K, C1 * K * K), $randn);
$b1 = array_fill(0, C1, 0.0);
$w2 = randnArr($SZ_W2, xavier(C1 * K * K, C2 * K * K), $randn);
$b2 = array_fill(0, C2, 0.0);
$w3 = randnArr($SZ_W3, xavier(FLAT, N_CLASS), $randn);
$b3 = array_fill(0, N_CLASS, 0.0);

// ====================================================================
// 3. Allocate GPU buffers
// ====================================================================
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
$b3Gpu = $backend->bufferAlloc(N_CLASS);

$xGpu    = $backend->bufferAlloc($SZ_X);
$c1Gpu   = $backend->bufferAlloc($SZ_C1);
$a1Gpu   = $backend->bufferAlloc($SZ_C1);
$m1Gpu   = $backend->bufferAlloc($SZ_C1);
$p1Gpu   = $backend->bufferAlloc($SZ_P1);
$arg1Gpu = $backend->bufferAlloc($SZ_P1);
$c2Gpu   = $backend->bufferAlloc($SZ_C2);
$a2Gpu   = $backend->bufferAlloc($SZ_C2);
$m2Gpu   = $backend->bufferAlloc($SZ_C2);
$p2Gpu   = $backend->bufferAlloc($SZ_P2);
$arg2Gpu = $backend->bufferAlloc($SZ_P2);
$z3Gpu   = $backend->bufferAlloc($BATCH * N_CLASS);
$dz3Gpu  = $backend->bufferAlloc($BATCH * N_CLASS);
$yGpu    = $backend->bufferAlloc($BATCH);
$lossGpu = $backend->bufferAlloc(1);

$dp2Gpu  = $backend->bufferAlloc($SZ_P2);
$da2Gpu  = $backend->bufferAlloc($SZ_C2);
$dc2Gpu  = $backend->bufferAlloc($SZ_C2);
$dp1Gpu  = $backend->bufferAlloc($SZ_P1);
$da1Gpu  = $backend->bufferAlloc($SZ_C1);
$dc1Gpu  = $backend->bufferAlloc($SZ_C1);
$dxGpu   = $backend->bufferAlloc($SZ_X);

$dw1Gpu = $backend->bufferAlloc($SZ_W1);
$db1Gpu = $backend->bufferAlloc(C1);
$dw2Gpu = $backend->bufferAlloc($SZ_W2);
$db2Gpu = $backend->bufferAlloc(C2);
$dw3Gpu = $backend->bufferAlloc($SZ_W3);
$db3Gpu = $backend->bufferAlloc(N_CLASS);

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

// ====================================================================
// 4. Training loop
// ====================================================================
$N = count($trainX);
$nBatches = (int) ceil($N / $BATCH);
$tStart = microtime(true);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $epochStart = microtime(true);
    $order = range(0, $N - 1);
    shuffle($order);

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * $BATCH;
        $end   = min($start + $BATCH, $N);
        $nReal = $end - $start;
        if ($nReal === 0) break;

        $xFlat = []; $yFlat = [];
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

        // Forward
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

        $backend->matmulDev($p2Gpu, $w3Gpu, $z3Gpu, $BATCH, FLAT, N_CLASS);
        $backend->addBiasDev($z3Gpu, $b3Gpu, $BATCH, N_CLASS);
        $backend->softmaxCeDev($z3Gpu, $yGpu, $dz3Gpu, $lossGpu, $BATCH, N_CLASS);

        // Backward
        $backend->zeroDev($dw3Gpu, $SZ_W3);
        $backend->matmulTnDev($p2Gpu, $dz3Gpu, $dw3Gpu, FLAT, $BATCH, N_CLASS);
        $backend->zeroDev($db3Gpu, N_CLASS);
        $backend->biasGradDev($dz3Gpu, $db3Gpu, $BATCH, N_CLASS);
        $backend->matmulNtDev($dz3Gpu, $w3Gpu, $dp2Gpu, $BATCH, N_CLASS, FLAT);
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

        // SGD
        $backend->sgdUpdateDev($w1Gpu, $dw1Gpu, $LR, $SZ_W1);
        $backend->sgdUpdateDev($b1Gpu, $db1Gpu, $LR, C1);
        $backend->sgdUpdateDev($w2Gpu, $dw2Gpu, $LR, $SZ_W2);
        $backend->sgdUpdateDev($b2Gpu, $db2Gpu, $LR, C2);
        $backend->sgdUpdateDev($w3Gpu, $dw3Gpu, $LR, $SZ_W3);
        $backend->sgdUpdateDev($b3Gpu, $db3Gpu, $LR, N_CLASS);
    }

    $backend->sync();
    printf("epoch %2d/%d   %.1fs\n", $epoch + 1, $EPOCHS, microtime(true) - $epochStart);
}

$trainSec = microtime(true) - $tStart;

// ====================================================================
// 5. Evaluate (same code as the audio example)
// ====================================================================
echo str_repeat('-', 60) . "\n";
echo "Evaluating " . count($testX) . " test samples...\n";

$backend->sync();
$w1T = $backend->bufferDownload($w1Gpu, $SZ_W1);
$b1T = $backend->bufferDownload($b1Gpu, C1);
$w2T = $backend->bufferDownload($w2Gpu, $SZ_W2);
$b2T = $backend->bufferDownload($b2Gpu, C2);
$w3T = $backend->bufferDownload($w3Gpu, $SZ_W3);
$b3T = $backend->bufferDownload($b3Gpu, N_CLASS);

$correct = 0;
foreach ($testX as $i => $img) {
    // Conv1
    $c1 = array_fill(0, C1 * C1_H * C1_W, 0.0);
    for ($oc = 0; $oc < C1; $oc++)
        for ($oh = 0; $oh < C1_H; $oh++)
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
    // Pool1
    $p1 = [];
    for ($c = 0; $c < C1; $c++)
        for ($oh = 0; $oh < P1_H; $oh++)
            for ($ow = 0; $ow < P1_W; $ow++) {
                $best = -1e30;
                for ($kh = 0; $kh < POOL; $kh++)
                    for ($kw = 0; $kw < POOL; $kw++) {
                        $v = $c1[$c * C1_H * C1_W + ($oh*POOL+$kh) * C1_W + ($ow*POOL+$kw)];
                        if ($v > $best) $best = $v;
                    }
                $p1[] = $best;
            }
    // Conv2
    $c2 = array_fill(0, C2 * C2_H * C2_W, 0.0);
    for ($oc = 0; $oc < C2; $oc++)
        for ($oh = 0; $oh < C2_H; $oh++)
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
    // Pool2
    $p2 = [];
    for ($c = 0; $c < C2; $c++)
        for ($oh = 0; $oh < P2_H; $oh++)
            for ($ow = 0; $ow < P2_W; $ow++) {
                $best = -1e30;
                for ($kh = 0; $kh < POOL; $kh++)
                    for ($kw = 0; $kw < POOL; $kw++) {
                        $v = $c2[$c * C2_H * C2_W + ($oh*POOL+$kh) * C2_W + ($ow*POOL+$kw)];
                        if ($v > $best) $best = $v;
                    }
                $p2[] = $best;
            }
    // Linear
    $logits = $b3T;
    for ($j = 0; $j < FLAT; $j++) {
        $v = $p2[$j];
        if ($v == 0.0) continue;
        $base = $j * N_CLASS;
        for ($k = 0; $k < N_CLASS; $k++) $logits[$k] += $v * $w3T[$base + $k];
    }
    $best = 0;
    for ($k = 1; $k < N_CLASS; $k++) if ($logits[$k] > $logits[$best]) $best = $k;
    if ($best === $testY[$i]) $correct++;
}

$acc = 100.0 * $correct / count($testX);

echo str_repeat('=', 60) . "\n";
printf("Test accuracy:   %.2f%% (%d / %d)\n", $acc, $correct, count($testX));
printf("Total training:  %.1f s\n", $trainSec);
printf("Per epoch:       %.1f s\n", $trainSec / $EPOCHS);