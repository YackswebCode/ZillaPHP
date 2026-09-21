<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Audio\MelFilterbank;
use ZillaPHP\Audio\MelSpectrogram;
use ZillaPHP\Audio\Wav;
use ZillaPHP\Core\Application;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;

Application::boot();

$backend = \ZillaPHP\Tensor\Tensor::backend();
if (!$backend instanceof NativeCpuBackend) {
    fwrite(STDERR, "Native backend required.\n");
    exit(1);
}

$SR  = 16000;
$FFT = 512;
$HOP = 160;
$N_MELS = 64;

$mel = new MelSpectrogram(
    backend:    $backend,
    sampleRate: $SR,
    fftSize:    $FFT,
    hopSize:    $HOP,
    nMels:      $N_MELS,
);

// ---- Inspect the filterbank once ----
$bank = new MelFilterbank($SR, $FFT, $N_MELS);
echo "Mel filterbank: [{$bank->nMels}, {$bank->nBins}]\n";
echo "First filter peak  : bin " . argmax($bank->matrix[0])  . "\n";
echo "Middle filter peak : bin " . argmax($bank->matrix[intdiv($N_MELS, 2)]) . "\n";
echo "Last filter peak   : bin " . argmax($bank->matrix[$N_MELS - 1]) . "\n\n";

// ---- Process test WAVs ----
$dir = __DIR__ . '/../data/audio/synthetic';

foreach (['sine_440hz_1s.wav', 'sine_3000hz_1s.wav', 'sweep_200_4000.wav'] as $fname) {
    $wav = Wav::fromFile("{$dir}/{$fname}");

    $t0 = microtime(true);
    $spec = $mel->fromWav($wav);
    $ms = (microtime(true) - $t0) * 1000;

    $dims = $spec->shape()->dims();
    $data = $spec->data();
    [$c, $m, $f] = $dims;

    printf("%-22s  [%d, %d, %d]  (%.1f ms)  min=%.2f  max=%.2f  mean=%.2f\n",
        $fname, $c, $m, $f, $ms,
        min($data), max($data), array_sum($data) / count($data)
    );

    // Show which mel band contains the most energy for the middle frame
    $midFrame = intdiv($f, 2);
    $peakMel  = 0;
    $peakVal  = -INF;
    for ($band = 0; $band < $m; $band++) {
        $v = $data[$band * $f + $midFrame];
        if ($v > $peakVal) { $peakVal = $v; $peakMel = $band; }
    }
    // Convert mel band index → approximate Hz center
    $melMin = MelFilterbank::hzToMel(0);
    $melMax = MelFilterbank::hzToMel($SR / 2);
    $melCenter = $melMin + ($melMax - $melMin) * ($peakMel + 1) / ($N_MELS + 1);
    $hzCenter = MelFilterbank::melToHz($melCenter);
    printf("%-22s  peak mel band=%d  ≈ %.0f Hz\n\n", '', $peakMel, $hzCenter);
}

function argmax(array $row): int
{
    $best = 0; $bv = -INF;
    foreach ($row as $i => $v) if ($v > $bv) { $bv = $v; $best = $i; }
    return $best;
}