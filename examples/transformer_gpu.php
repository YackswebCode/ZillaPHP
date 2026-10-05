<?php

declare(strict_types=1);

/**
 * GPU Transformer training — v1.
 *
 * Tiny single-head char-level LM. No LayerNorm (v1 simplification).
 * Proves the full GPU Transformer training loop works end-to-end.
 */

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Hardware\CUDA\CudaBackend;

Application::boot(cuda: true);

$backend = \ZillaPHP\Tensor\Tensor::backend();
if (!$backend instanceof CudaBackend || !$backend->isCuda()) {
    fwrite(STDERR, "CUDA backend required.\n");
    exit(1);
}

// ====================================================================
// Config
// ====================================================================
$SEQ      = (int)   (getenv('SEQ')      ?: 32);
$D_MODEL  = (int)   (getenv('D_MODEL')  ?: 32);
$D_FF     = (int)   (getenv('D_FF')     ?: 64);
$STEPS    = (int)   (getenv('STEPS')    ?: 500);
$LR       = (float) (getenv('LR')       ?: 0.01);

$textFile = __DIR__ . '/../data/shakespeare.txt';
if (!is_file($textFile)) {
    fwrite(STDERR, "Shakespeare not found. Run: php scripts/download_shakespeare.php\n");
    exit(1);
}
$text = substr(file_get_contents($textFile), 0, 50000);

// Char-level vocab
$chars = array_values(array_unique(str_split($text)));
sort($chars);
$vocabSize = count($chars);
$charToId  = array_flip($chars);
$ids = array_map(fn($c) => $charToId[$c], str_split($text));
$nTokens = count($ids);

$scale = 1.0 / sqrt($D_MODEL);

// ====================================================================
// Init weights (Xavier)
// ====================================================================
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

$embW = randnArr($vocabSize * $D_MODEL, xavier($vocabSize, $D_MODEL));
$posW = randnArr($SEQ       * $D_MODEL, xavier($SEQ, $D_MODEL));

$wq = randnArr($D_MODEL * $D_MODEL, xavier($D_MODEL, $D_MODEL));
$wk = randnArr($D_MODEL * $D_MODEL, xavier($D_MODEL, $D_MODEL));
$wv = randnArr($D_MODEL * $D_MODEL, xavier($D_MODEL, $D_MODEL));
$wo = randnArr($D_MODEL * $D_MODEL, xavier($D_MODEL, $D_MODEL));

$w1 = randnArr($D_MODEL * $D_FF, xavier($D_MODEL, $D_FF));
$b1 = array_fill(0, $D_FF, 0.0);
$w2 = randnArr($D_FF * $D_MODEL, xavier($D_FF, $D_MODEL));
$b2 = array_fill(0, $D_MODEL, 0.0);

$wlm = randnArr($D_MODEL * $vocabSize, xavier($D_MODEL, $vocabSize));
$blm = array_fill(0, $vocabSize, 0.0);

// ====================================================================
// Allocate GPU buffers — ONCE
// ====================================================================
echo "Allocating GPU buffers...\n";

$SZ_EMB   = $vocabSize * $D_MODEL;
$SZ_POS   = $SEQ * $D_MODEL;
$SZ_X     = $SEQ * $D_MODEL;          // X = emb + pos
$SZ_QKV   = $SEQ * $D_MODEL;
$SZ_S     = $SEQ * $SEQ;
$SZ_FF    = $SEQ * $D_FF;
$SZ_LOGI  = $SEQ * $vocabSize;

// ---- Weights ----
$embGpu  = $backend->bufferAlloc($SZ_EMB);
$posGpu  = $backend->bufferAlloc($SZ_POS);
$wqGpu   = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$wkGpu   = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$wvGpu   = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$woGpu   = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$w1Gpu   = $backend->bufferAlloc($D_MODEL * $D_FF);
$b1Gpu   = $backend->bufferAlloc($D_FF);
$w2Gpu   = $backend->bufferAlloc($D_FF * $D_MODEL);
$b2Gpu   = $backend->bufferAlloc($D_MODEL);
$wlmGpu  = $backend->bufferAlloc($D_MODEL * $vocabSize);
$blmGpu  = $backend->bufferAlloc($vocabSize);

// ---- Activations ----
$xGpu      = $backend->bufferAlloc($SZ_X);
$qGpu      = $backend->bufferAlloc($SZ_QKV);
$kTmpGpu   = $backend->bufferAlloc($D_MODEL * $SEQ);   // K transposed [D, SEQ]
$vGpu      = $backend->bufferAlloc($SZ_QKV);
$sGpu      = $backend->bufferAlloc($SZ_S);             // attention scores (pre-softmax)
$pGpu      = $backend->bufferAlloc($SZ_S);             // post-softmax
$oGpu      = $backend->bufferAlloc($SZ_QKV);           // attention output
$y1Gpu     = $backend->bufferAlloc($SZ_X);             // X + O (residual)
$f1Gpu     = $backend->bufferAlloc($SZ_FF);            // Y1 @ W1 + b1
$f1reluGpu = $backend->bufferAlloc($SZ_FF);            // ReLU(F1)
$m1FfGpu   = $backend->bufferAlloc($SZ_FF);            // ReLU mask (allocated once)
$f2Gpu     = $backend->bufferAlloc($SZ_X);             // F1_relu @ W2 + b2
$y2Gpu     = $backend->bufferAlloc($SZ_X);             // Y1 + F2
$logGpu    = $backend->bufferAlloc($SZ_LOGI);          // Y2 @ Wlm + blm

// ---- Backward temporaries ----
$dLogGpu    = $backend->bufferAlloc($SZ_LOGI);
$dY2Gpu     = $backend->bufferAlloc($SZ_X);
$dF2Gpu     = $backend->bufferAlloc($SZ_X);
$dF1Gpu     = $backend->bufferAlloc($SZ_FF);
$dF1ReluGpu = $backend->bufferAlloc($SZ_FF);
$dY1Gpu     = $backend->bufferAlloc($SZ_X);
$dOGpu      = $backend->bufferAlloc($SZ_QKV);
$dPGpu      = $backend->bufferAlloc($SZ_S);
$dSGpu      = $backend->bufferAlloc($SZ_S);
$dQGpu      = $backend->bufferAlloc($SZ_QKV);
$dKTmpGpu   = $backend->bufferAlloc($D_MODEL * $SEQ);
$dKGpu      = $backend->bufferAlloc($SZ_QKV);
$dVGpu      = $backend->bufferAlloc($SZ_QKV);
$dXGpu      = $backend->bufferAlloc($SZ_X);

// ---- Gradient accumulators ----
$dEmbGpu = $backend->bufferAlloc($SZ_EMB);
$dPosGpu = $backend->bufferAlloc($SZ_POS);
$dWqGpu  = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$dWkGpu  = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$dWvGpu  = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$dWoGpu  = $backend->bufferAlloc($D_MODEL * $D_MODEL);
$dW1Gpu  = $backend->bufferAlloc($D_MODEL * $D_FF);
$db1Gpu  = $backend->bufferAlloc($D_FF);
$dW2Gpu  = $backend->bufferAlloc($D_FF * $D_MODEL);
$db2Gpu  = $backend->bufferAlloc($D_MODEL);
$dWlmGpu = $backend->bufferAlloc($D_MODEL * $vocabSize);
$dblmGpu = $backend->bufferAlloc($vocabSize);

// ---- Targets + loss ----
$yInGpu  = $backend->bufferAlloc($SEQ);
$lossGpu = $backend->bufferAlloc(1);

// ---- Upload initial weights ----
$backend->bufferUpload($embGpu, $embW);
$backend->bufferUpload($posGpu, $posW);
$backend->bufferUpload($wqGpu,  $wq);
$backend->bufferUpload($wkGpu,  $wk);
$backend->bufferUpload($wvGpu,  $wv);
$backend->bufferUpload($woGpu,  $wo);
$backend->bufferUpload($w1Gpu,  $w1);
$backend->bufferUpload($b1Gpu,  $b1);
$backend->bufferUpload($w2Gpu,  $w2);
$backend->bufferUpload($b2Gpu,  $b2);
$backend->bufferUpload($wlmGpu, $wlm);
$backend->bufferUpload($blmGpu, $blm);

echo "Backend: " . $backend->name() . "\n";
echo "Transformer: dim=$D_MODEL, head=1, seq=$SEQ, ffn=$D_FF, vocab=$vocabSize\n";
echo "Training on " . $nTokens . " tokens from Shakespeare\n";
echo str_repeat('=', 60) . "\n";

// ====================================================================
// Cache the embeddings on CPU side (avoids downloading per step)
// ====================================================================
$cpuEmb = $embW;   // We keep a CPU copy in sync with the GPU buffer

// ====================================================================
// Training loop
// ====================================================================
$tStart = microtime(true);
$runningLoss = 0.0;

for ($step = 0; $step < $STEPS; $step++) {
    // ---- Sample a sequence ----
    $start = mt_rand(0, $nTokens - $SEQ - 1);
    $inputIds  = array_slice($ids, $start, $SEQ);
    $targetIds = array_slice($ids, $start + 1, $SEQ);

    // ---- Build X = emb[inputIds] + pos[0..SEQ-1] on CPU ----
    // Use the local $cpuEmb and $posW caches (kept in sync with GPU updates)
    $xFlat = [];
    for ($t = 0; $t < $SEQ; $t++) {
        $tok = $inputIds[$t];
        for ($d = 0; $d < $D_MODEL; $d++) {
            $xFlat[] = $cpuEmb[$tok * $D_MODEL + $d] + $posW[$t * $D_MODEL + $d];
        }
    }
    $backend->bufferUpload($xGpu, $xFlat);

    // Targets as float
    $yFlat = array_map('floatval', $targetIds);
    $backend->bufferUpload($yInGpu, $yFlat);

    // ---- Zero gradients + loss ----
    $backend->zeroDev($lossGpu, 1);
    foreach ([
        [$dEmbGpu, $SZ_EMB], [$dPosGpu, $SZ_POS],
        [$dWqGpu,  $D_MODEL*$D_MODEL], [$dWkGpu, $D_MODEL*$D_MODEL],
        [$dWvGpu,  $D_MODEL*$D_MODEL], [$dWoGpu, $D_MODEL*$D_MODEL],
        [$dW1Gpu,  $D_MODEL*$D_FF],    [$db1Gpu, $D_FF],
        [$dW2Gpu,  $D_FF*$D_MODEL],    [$db2Gpu, $D_MODEL],
        [$dWlmGpu, $D_MODEL*$vocabSize],[$dblmGpu, $vocabSize],
    ] as [$id, $sz]) {
        $backend->zeroDev($id, $sz);
    }

    // ================================================================
    // FORWARD
    // ================================================================

    // Q = X @ Wq
    $backend->matmulDev($xGpu, $wqGpu, $qGpu, $SEQ, $D_MODEL, $D_MODEL);

    // K = X @ Wk  (into vGpu temporarily)
    $backend->matmulDev($xGpu, $wkGpu, $vGpu, $SEQ, $D_MODEL, $D_MODEL);
    $backend->sync();
    $kFlat = $backend->bufferDownload($vGpu, $SZ_QKV);

    // K^T — transpose on CPU (small: SEQ × D_MODEL)
    $kT = [];
    for ($d = 0; $d < $D_MODEL; $d++)
        for ($t = 0; $t < $SEQ; $t++) $kT[] = $kFlat[$t * $D_MODEL + $d];
    $backend->bufferUpload($kTmpGpu, $kT);

    // V = X @ Wv  (overwrite vGpu)
    $backend->matmulDev($xGpu, $wvGpu, $vGpu, $SEQ, $D_MODEL, $D_MODEL);

    // S = Q @ K^T  [SEQ, SEQ]
    $backend->matmulDev($qGpu, $kTmpGpu, $sGpu, $SEQ, $D_MODEL, $SEQ);
    $backend->scaleDev($sGpu, $scale, $SEQ * $SEQ);

    // P = causal_softmax(S) — in place
    $backend->causalSoftmaxDev($sGpu, $SEQ, $SEQ);
    $backend->sync();
    $pFlat = $backend->bufferDownload($sGpu, $SZ_S);
    $backend->bufferUpload($pGpu, $pFlat);

    // O = P @ V  [SEQ, D_MODEL]
    $backend->matmulDev($pGpu, $vGpu, $oGpu, $SEQ, $SEQ, $D_MODEL);

    // ---- Y1 = X + O (residual) ----
    $backend->sync();
    $oFlat = $backend->bufferDownload($oGpu, $SZ_QKV);
    $y1Flat = [];
    for ($i = 0; $i < $SZ_X; $i++) $y1Flat[] = $xFlat[$i] + $oFlat[$i];
    $backend->bufferUpload($y1Gpu, $y1Flat);

    // ---- F1 = Y1 @ W1 + b1 ----
    $backend->matmulDev($y1Gpu, $w1Gpu, $f1Gpu, $SEQ, $D_MODEL, $D_FF);
    $backend->addBiasDev($f1Gpu, $b1Gpu, $SEQ, $D_FF);

    // ---- F1_relu = ReLU(F1); save mask ----
    $backend->reluFwdDev($f1Gpu, $f1reluGpu, $m1FfGpu, $SZ_FF);

    // ---- F2 = F1_relu @ W2 + b2 ----
    $backend->matmulDev($f1reluGpu, $w2Gpu, $f2Gpu, $SEQ, $D_FF, $D_MODEL);
    $backend->addBiasDev($f2Gpu, $b2Gpu, $SEQ, $D_MODEL);

    // ---- Y2 = Y1 + F2 ----
    $backend->sync();
    $f2Flat = $backend->bufferDownload($f2Gpu, $SZ_X);
    $y2Flat = [];
    for ($i = 0; $i < $SZ_X; $i++) $y2Flat[] = $y1Flat[$i] + $f2Flat[$i];
    $backend->bufferUpload($y2Gpu, $y2Flat);

    // ---- Logits = Y2 @ Wlm + blm ----
    $backend->matmulDev($y2Gpu, $wlmGpu, $logGpu, $SEQ, $D_MODEL, $vocabSize);
    $backend->addBiasDev($logGpu, $blmGpu, $SEQ, $vocabSize);

    // ================================================================
    // LOSS
    // ================================================================
    $backend->softmaxCeDev($logGpu, $yInGpu, $dLogGpu, $lossGpu, $SEQ, $vocabSize);
    $backend->sync();

    $lossVal = $backend->bufferDownload($lossGpu, 1)[0] / $SEQ;
    $runningLoss += $lossVal;

    // ================================================================
    // BACKWARD
    // ================================================================

    // --- Through LM head ---
    // dW_lm = Y2^T @ dLogits
    $backend->matmulTnDev($y2Gpu, $dLogGpu, $dWlmGpu, $D_MODEL, $SEQ, $vocabSize);
    $backend->biasGradDev($dLogGpu, $dblmGpu, $SEQ, $vocabSize);
    // dY2 = dLogits @ W_lm^T
    $backend->matmulNtDev($dLogGpu, $wlmGpu, $dY2Gpu, $SEQ, $vocabSize, $D_MODEL);

    // --- Through FFN ---
    // dF2 = dY2 (identity through Y2 = Y1 + F2)
    // dW2 = F1_relu^T @ dF2
    $backend->matmulTnDev($f1reluGpu, $dY2Gpu, $dW2Gpu, $D_FF, $SEQ, $D_MODEL);
    $backend->biasGradDev($dY2Gpu, $db2Gpu, $SEQ, $D_MODEL);
    // dF1_relu = dF2 @ W2^T
    $backend->matmulNtDev($dY2Gpu, $w2Gpu, $dF1ReluGpu, $SEQ, $D_MODEL, $D_FF);
    // dF1 = dF1_relu * mask
    $backend->reluBwdDev($dF1ReluGpu, $m1FfGpu, $dF1Gpu, $SZ_FF);
    // dW1 = Y1^T @ dF1
    $backend->matmulTnDev($y1Gpu, $dF1Gpu, $dW1Gpu, $D_MODEL, $SEQ, $D_FF);
    $backend->biasGradDev($dF1Gpu, $db1Gpu, $SEQ, $D_FF);
    // dY1_ffn = dF1 @ W1^T
    $backend->matmulNtDev($dF1Gpu, $w1Gpu, $dY1Gpu, $SEQ, $D_FF, $D_MODEL);

    // --- Combine residuals into dY1 ---
    $backend->sync();
    $dy1fFlat = $backend->bufferDownload($dY1Gpu, $SZ_X);
    $dy2Flat  = $backend->bufferDownload($dY2Gpu, $SZ_X);
    $dy1Total = [];
    for ($i = 0; $i < $SZ_X; $i++) $dy1Total[] = $dy1fFlat[$i] + $dy2Flat[$i];
    $backend->bufferUpload($dY1Gpu, $dy1Total);

    // --- Through attention ---
    // dO = dY1
    $backend->bufferUpload($dOGpu, $dy1Total);

    // dP = dO @ V^T
    $backend->matmulNtDev($dOGpu, $vGpu, $dPGpu, $SEQ, $D_MODEL, $SEQ);
    // dV = P^T @ dO
    $backend->matmulTnDev($pGpu, $dOGpu, $dVGpu, $SEQ, $SEQ, $D_MODEL);
    // dS = softmax_bwd(P, dP)
    $backend->softmaxRowsBwdDev($pGpu, $dPGpu, $dSGpu, $SEQ, $SEQ);
    // Scale back: dS_prescaled = dS * scale
    $backend->scaleDev($dSGpu, $scale, $SEQ * $SEQ);

    // dQ = dS @ K
    $backend->matmulNtDev($dSGpu, $kTmpGpu, $dQGpu, $SEQ, $SEQ, $D_MODEL);
    // dK = dS^T @ Q
    $backend->matmulTnDev($dSGpu, $qGpu, $dKGpu, $SEQ, $SEQ, $D_MODEL);

    // --- Weight gradients for QKV ---
    $backend->matmulTnDev($xGpu, $dQGpu, $dWqGpu, $D_MODEL, $SEQ, $D_MODEL);
    $backend->matmulTnDev($xGpu, $dKGpu, $dWkGpu, $D_MODEL, $SEQ, $D_MODEL);
    $backend->matmulTnDev($xGpu, $dVGpu, $dWvGpu, $D_MODEL, $SEQ, $D_MODEL);

    // ================================================================
    // SGD UPDATES
    // ================================================================
    $backend->sgdUpdateDev($wqGpu,  $dWqGpu,  $LR, $D_MODEL * $D_MODEL);
    $backend->sgdUpdateDev($wkGpu,  $dWkGpu,  $LR, $D_MODEL * $D_MODEL);
    $backend->sgdUpdateDev($wvGpu,  $dWvGpu,  $LR, $D_MODEL * $D_MODEL);
    $backend->sgdUpdateDev($w1Gpu,  $dW1Gpu,  $LR, $D_MODEL * $D_FF);
    $backend->sgdUpdateDev($b1Gpu,  $db1Gpu,  $LR, $D_FF);
    $backend->sgdUpdateDev($w2Gpu,  $dW2Gpu,  $LR, $D_FF * $D_MODEL);
    $backend->sgdUpdateDev($b2Gpu,  $db2Gpu,  $LR, $D_MODEL);
    $backend->sgdUpdateDev($wlmGpu, $dWlmGpu, $LR, $D_MODEL * $vocabSize);
    $backend->sgdUpdateDev($blmGpu, $dblmGpu, $LR, $vocabSize);

    if (($step + 1) % 25 === 0) {
        $avgLoss = $runningLoss / 25;
        printf("step %4d/%d   loss=%.4f\n", $step + 1, $STEPS, $avgLoss);
        $runningLoss = 0.0;
    }
}

$totalTime = microtime(true) - $tStart;

echo str_repeat('=', 60) . "\n";
printf("Total:    %.1f s\n", $totalTime);
printf("Per step: %.4f s\n", $totalTime / $STEPS);