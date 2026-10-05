<?php

declare(strict_types=1);

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

// ---- Small fixed input ----
$C = 1; $H = 4; $W = 4;
$outC = 1; $kH = 3; $kW = 3; $stride = 1; $padding = 0;
$outH = intdiv($H + 2 * $padding - $kH, $stride) + 1;   // = 2
$outW = intdiv($W + 2 * $padding - $kW, $stride) + 1;   // = 2

$input = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
];
$weight = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0];
$bias = [0.0];

// ---- Expected (pure PHP) ----
$refOut = [];
for ($oh = 0; $oh < $outH; $oh++) {
    for ($ow = 0; $ow < $outW; $ow++) {
        $sum = $bias[0];
        for ($kh = 0; $kh < $kH; $kh++) {
            for ($kw = 0; $kw < $kW; $kw++) {
                $ih = $oh * $stride - $padding + $kh;
                $iw = $ow * $stride - $padding + $kw;
                $sum += $input[$ih * $W + $iw] * $weight[$kh * $kW + $kw];
            }
        }
        $refOut[] = $sum;
    }
}

// ---- GPU conv ----
$inId  = $backend->bufferAlloc($C * $H * $W);
$wId   = $backend->bufferAlloc($outC * $C * $kH * $kW);
$bId   = $backend->bufferAlloc($outC);
$outId = $backend->bufferAlloc($outC * $outH * $outW);

$backend->bufferUpload($inId, $input);
$backend->bufferUpload($wId, $weight);
$backend->bufferUpload($bId, $bias);

$backend->conv2dFwdDev($inId, $wId, $bId, $outId, $C, $H, $W, $outC, $outH, $outW, $kH, $kW, $stride, $padding);
$backend->sync();
$out = $backend->bufferDownload($outId, $outC * $outH * $outW);

$maxErr = 0.0;
foreach ($refOut as $i => $v) $maxErr = max($maxErr, abs($v - $out[$i]));

printf("Input:    %dx%dx%d\n", $C, $H, $W);
printf("Kernel:   %dx%d, stride=%d, padding=%d\n", $kH, $kW, $stride, $padding);
printf("Output:   %dx%dx%d\n", $outC, $outH, $outW);
printf("Expected: %s\n", json_encode($refOut));
printf("Got:      %s\n", json_encode($out));
printf("Max err:  %.8f\n", $maxErr);
echo ($maxErr < 1e-4) ? "CONV2D OK\n" : "CONV2D FAILED\n";

// ---- Cleanup ----
$backend->bufferFree($inId);
$backend->bufferFree($wId);
$backend->bufferFree($bId);
$backend->bufferFree($outId);