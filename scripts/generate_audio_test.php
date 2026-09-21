<?php

declare(strict_types=1);

/**
 * Generate a small set of synthetic WAV files for testing the audio
 * pipeline without downloading a real dataset.
 *
 *   - sine sweeps (frequency goes from low to high)
 *   - pure tones at fixed frequencies
 *   - white noise
 *   - silence
 *
 * All files are 16-bit PCM mono at 16000 Hz — the standard for
 * speech and the Speech Commands dataset.
 */

$outDir = __DIR__ . '/../data/audio/synthetic';
if (!is_dir($outDir)) mkdir($outDir, 0777, true);

const SAMPLE_RATE = 16000;

function writeWav(string $path, array $samples, int $sampleRate = SAMPLE_RATE): void
{
    $numSamples = count($samples);
    $dataBytes  = $numSamples * 2;
    $chunkSize  = 36 + $dataBytes;

    $header = 'RIFF' . pack('V', $chunkSize) . 'WAVE';
    $header .= 'fmt ' . pack('V', 16)          // subchunk1 size
             . pack('v', 1)                    // PCM
             . pack('v', 1)                    // mono
             . pack('V', $sampleRate)
             . pack('V', $sampleRate * 2)      // byte rate
             . pack('v', 2)                    // block align
             . pack('v', 16);                  // bits per sample
    $header .= 'data' . pack('V', $dataBytes);

    $pcm = '';
    foreach ($samples as $s) {
        $s = max(-1.0, min(1.0, $s));
        $i16 = (int) round($s * 32767);
        $pcm .= pack('v', $i16 & 0xFFFF);
    }

    file_put_contents($path, $header . $pcm);
}

function sine(float $freq, float $duration, int $sampleRate = SAMPLE_RATE): array
{
    $n = (int) ($duration * $sampleRate);
    $out = [];
    for ($i = 0; $i < $n; $i++) {
        $out[] = 0.6 * sin(2 * M_PI * $freq * $i / $sampleRate);
    }
    return $out;
}

function sweep(float $f0, float $f1, float $duration, int $sampleRate = SAMPLE_RATE): array
{
    $n = (int) ($duration * $sampleRate);
    $out = [];
    for ($i = 0; $i < $n; $i++) {
        $t = $i / $sampleRate;
        $f = $f0 + ($f1 - $f0) * ($t / $duration);
        $out[] = 0.6 * sin(2 * M_PI * $f * $t);
    }
    return $out;
}

function noise(float $duration, int $sampleRate = SAMPLE_RATE): array
{
    $n = (int) ($duration * $sampleRate);
    $out = [];
    for ($i = 0; $i < $n; $i++) {
        $out[] = 0.4 * (mt_rand() / mt_getrandmax() * 2 - 1);
    }
    return $out;
}

function silence(float $duration, int $sampleRate = SAMPLE_RATE): array
{
    return array_fill(0, (int) ($duration * $sampleRate), 0.0);
}

// ---- Generate ----
$tests = [
    'sine_440hz_1s.wav'    => sine(440.0, 1.0),
    'sine_1000hz_1s.wav'   => sine(1000.0, 1.0),
    'sine_3000hz_1s.wav'   => sine(3000.0, 1.0),
    'sweep_200_4000.wav'   => sweep(200.0, 4000.0, 1.5),
    'noise_1s.wav'         => noise(1.0),
    'silence_1s.wav'       => silence(1.0),
];

foreach ($tests as $name => $samples) {
    writeWav("{$outDir}/{$name}", $samples);
    printf("Wrote %-22s  %6d samples  %5.2f s\n",
        $name, count($samples), count($samples) / SAMPLE_RATE);
}

echo "\nDone. Files in {$outDir}\n";