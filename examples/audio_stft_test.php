<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Audio\Wav;
use ZillaPHP\Core\Application;

Application::boot();

$backend = \ZillaPHP\Tensor\Tensor::backend();

if (!method_exists($backend, 'stftMagnitude')) {
    fwrite(STDERR, "Backend does not expose stftMagnitude — rebuild native .so.\n");
    exit(1);
}

// ---- Config ----
$FFT = 512;
$HOP = 160;   // 10 ms at 16 kHz

// ---- Load a test WAV ----
$wav  = Wav::fromFile(__DIR__ . '/../data/audio/synthetic/sine_440hz_1s.wav');
$wave = $wav->data;

echo "Loaded: {$wav->summary()}\n";
printf("FFT=%d  Hop=%d\n\n", $FFT, $HOP);

// ---- Hann window ----
$t0 = microtime(true);
$window = $backend->hannWindow($FFT);
$tWin = microtime(true) - $t0;

printf("Hann window: N=%d  w[0]=%.4f  w[%d]=%.4f  w[%d]=%.4f  (%.2f ms)\n",
    $FFT,
    $window[0],
    intdiv($FFT, 2), $window[intdiv($FFT, 2)],
    $FFT - 1, $window[$FFT - 1],
    1000 * $tWin
);

// ---- STFT ----
$t0 = microtime(true);
[$mag, $nBins, $nFrames] = $backend->stftMagnitude($wave, $window, $FFT, $HOP);
$tStft = microtime(true) - $t0;

printf("STFT: [%d bins, %d frames]  (%.1f ms)\n\n", $nBins, $nFrames, 1000 * $tStft);

// ---- Locate peak bin in a middle frame ----
$midFrame = intdiv($nFrames, 2);
$peakBin  = 0;
$peakVal  = -1.0;
for ($b = 0; $b < $nBins; $b++) {
    $v = $mag[$b * $nFrames + $midFrame];
    if ($v > $peakVal) { $peakVal = $v; $peakBin = $b; }
}

$binHz    = $peakBin * $wav->sampleRate / $FFT;
$expected = (int) round(440.0 * $FFT / $wav->sampleRate);
$expHz    = $expected * $wav->sampleRate / $FFT;

printf("Peak in middle frame: bin %d  (%.1f Hz)  magnitude=%.2f\n",
    $peakBin, $binHz, $peakVal);
printf("Expected peak for 440 Hz sine: bin %d (%.1f Hz)\n\n",
    $expected, $expHz);

if ($peakBin === $expected) {
    echo "STFT is correct — peak matches expected bin.\n\n";
} else {
    echo "WARNING: peak bin differs from expected — inspect.\n\n";
}

// ---- ASCII magnitude spectrum ----
echo "ASCII magnitude spectrum (frame {$midFrame}):\n";
for ($b = 0; $b < min(64, $nBins); $b++) {
    $v    = $mag[$b * $nFrames + $midFrame];
    $bars = (int) round($v * 2);
    printf("bin %3d (%5.0f Hz): %s\n",
        $b,
        $b * $wav->sampleRate / $FFT,
        str_repeat('#', $bars)
    );
}