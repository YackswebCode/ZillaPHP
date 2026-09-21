<?php
require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Audio\Wav;
use ZillaPHP\Core\Application;

Application::boot(cuda: true);
$backend = \ZillaPHP\Tensor\Tensor::backend();
echo "Backend: " . $backend->name() . "\n";

$wavFile = __DIR__ . '/../data/audio/synthetic/sine_440hz_1s.wav';

// Generate the test WAV if it doesn't exist
if (!is_file($wavFile)) {
    echo "Generating test WAV...\n";
    $sr = 16000;
    $n  = $sr;
    $samples = [];
    for ($i = 0; $i < $n; $i++) {
        $samples[] = 0.6 * sin(2 * M_PI * 440 * $i / $sr);
    }
    $dataBytes = $n * 2;
    $header  = 'RIFF' . pack('V', 36 + $dataBytes) . 'WAVE';
    $header .= 'fmt ' . pack('V', 16) . pack('v', 1) . pack('v', 1)
             . pack('V', $sr) . pack('V', $sr * 2) . pack('v', 2) . pack('v', 16);
    $header .= 'data' . pack('V', $dataBytes);
    $pcm = '';
    foreach ($samples as $s) {
        $pcm .= pack('v', ((int) round($s * 32767)) & 0xFFFF);
    }
    @mkdir(dirname($wavFile), 0777, true);
    file_put_contents($wavFile, $header . $pcm);
}

$wav    = Wav::fromFile($wavFile);
$window = $backend->hannWindow(512);

$t0 = microtime(true);
[$mag, $nBins, $nFrames] = $backend->stftMagnitude($wav->data, $window, 512, 160);
$ms = (microtime(true) - $t0) * 1000;

printf("STFT: [%d bins, %d frames]  (%.1f ms)\n", $nBins, $nFrames, $ms);

$midFrame = intdiv($nFrames, 2);
$peakBin = 0; $peakVal = -1.0;
for ($b = 0; $b < $nBins; $b++) {
    $v = $mag[$b * $nFrames + $midFrame];
    if ($v > $peakVal) { $peakVal = $v; $peakBin = $b; }
}
printf("Peak bin: %d (%.1f Hz)\n", $peakBin, $peakBin * 16000 / 512);
echo ($peakBin === 14) ? "STFT OK\n" : "STFT MISMATCH\n";