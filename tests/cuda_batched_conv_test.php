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

// Small batch, single channel, 4x4 image
$B = 2; $C = 1; $H = 4; $W = 4;
$outC = 1; $kH = 3; $kW = 3; $stride = 1; $pad = 0;
$outH = 2; $outW = 2;

$input = [
    // batch 0
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
    // batch 1
    16.0, 15.0, 14.0, 13.0,
    12.0, 11.0, 10.0, 9.0,
    8.0, 7.0, 6.0, 5.0,
    4.0, 3.0, 2.0, 1.0,
];
$weight = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0];
$bias = [0.0];

// Reference (CPU)
$refOut = [];
for ($b = 0; $b < $B; $b++) {
    for ($oh = 0; $oh < $outH; $oh++) {
        for ($ow = 0; $ow < $outW; $ow++) {
            $sum = $bias[0];
            for ($kh = 0; $kh < $kH; $kh++) {
                for ($kw = 0; $kw < $kW; $kw++) {
                    $ih = $oh * $stride - $pad + $kh;
                    $iw = $ow * $stride - $pad + $kw;
                    $sum += $input[$b * $H * $W + $ih * $W + $iw] * $weight[$kh * $kW + $kw];
                }
            }
            $refOut[] = $sum;
        }
    }
}

$inId  = $backend->bufferAlloc($B * $C * $H * $W);
$wId   = $backend->bufferAlloc($outC * $C * $kH * $kW);
$bId   = $backend->bufferAlloc($outC);
$outId = $backend->bufferAlloc($B * $outC * $outH * $outW);

$backend->bufferUpload($inId, $input);
$backend->bufferUpload($wId, $weight);
$backend->bufferUpload($bId, $bias);

$backend->conv2dBatchedFwdDev(
    $inId, $wId, $bId, $outId,
    $B, $C, $H, $W, $outC, $outH, $outW,
    $kH, $kW, $stride, $pad
);
$backend->sync();
$out = $backend->bufferDownload($outId, $B * $outC * $outH * $outW);

$maxErr = 0.0;
foreach ($refOut as $i => $v) $maxErr = max($maxErr, abs($v - $out[$i]));

printf("Batched Conv2D: B=%d C=%d %dx%d -> %d channels %dx%d\n", $B, $C, $H, $W, $outC, $outH, $outW);
printf("Expected: %s\n", json_encode($refOut));
printf("Got:      %s\n", json_encode($out));
printf("Max err:  %.8f\n", $maxErr);
echo ($maxErr < 1e-4) ? "BATCHED CONV2D OK\n" : "BATCHED CONV2D FAILED\n";

$backend->bufferFree($inId);
$backend->bufferFree($wId);
$backend->bufferFree($bId);
$backend->bufferFree($outId);