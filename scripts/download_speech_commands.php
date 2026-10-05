<?php

declare(strict_types=1);

/**
 * Downloads and extracts Google Speech Commands v2.
 *
 *   ~105,000 one-second WAV clips at 16 kHz mono
 *   35 word classes + _silence_ + _unknown_
 *   Total download: ~2.3 GB compressed, ~2.6 GB extracted
 *
 * Usage:
 *   php scripts/download_speech_commands.php
 *   php scripts/download_speech_commands.php --subset   # only 10 core words
 */

$baseDir = __DIR__ . '/../data/speech_commands';
$url     = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz';
$tarball = "{$baseDir}/speech_commands_v0.02.tar.gz";

if (!is_dir($baseDir)) mkdir($baseDir, 0777, true);

$subset = in_array('--subset', $argv, true);

// The 10 "core" words most papers use
$coreWords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'];

// ---- Download (if missing) ----
if (!is_file($tarball)) {
    echo "Downloading Speech Commands v2 (~2.3 GB)...\n";
    echo "This may take 5-15 minutes depending on your connection.\n\n";

    $fp = fopen($tarball, 'wb');
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_FILE, $fp);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_PROGRESSFUNCTION, function ($res, $dl, $total) {
        static $last = 0;
        if ($total > 0 && time() - $last >= 2) {
            printf("  %.1f%% (%.0f MB / %.0f MB)\n",
                100.0 * $dl / $total,
                $dl / 1048576,
                $total / 1048576
            );
            $last = time();
        }
        return 0;
    });
    curl_setopt($ch, CURLOPT_NOPROGRESS, false);
    curl_exec($ch);
    curl_close($ch);
    fclose($fp);
    echo "\nDownload complete.\n\n";
} else {
    echo "Tarball already present.\n\n";
}

// ---- Extract ----
echo "Extracting" . ($subset ? " (core words only)" : "") . "...\n";

if ($subset) {
    // Extract only the 10 core word directories
    $args = '';
    foreach ($coreWords as $w) {
        $args .= ' ' . escapeshellarg("{$w}/*");
    }
    // Plus the validation file (list of test samples)
    $cmd = "tar -xzf " . escapeshellarg($tarball)
         . " -C " . escapeshellarg($baseDir)
         . " {$args} validation_list.txt testing_list.txt";
    echo "Running (this may take a few minutes)...\n";
    passthru($cmd, $code);

    if ($code !== 0) {
        // tar with patterns doesn't always work cleanly; fall back to full extract
        echo "Pattern extraction failed; doing full extract...\n";
        passthru("tar -xzf " . escapeshellarg($tarball)
               . " -C " . escapeshellarg($baseDir), $code);
    }
} else {
    passthru("tar -xzf " . escapeshellarg($tarball)
           . " -C " . escapeshellarg($baseDir), $code);
}

if ($code !== 0) {
    fwrite(STDERR, "Extraction failed (exit {$code}).\n");
    exit(1);
}

// ---- Summary ----
echo "\n=== Dataset summary ===\n";

$words = $subset ? $coreWords : array_merge(
    $coreWords,
    ['bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
     'backward', 'forward', 'follow', 'learn', 'visual', '_silence_']
);

$total = 0;
foreach ($words as $w) {
    $dir = "{$baseDir}/{$w}";
    if (!is_dir($dir)) continue;
    $count = count(glob("{$dir}/*.wav"));
    printf("  %-12s %5d\n", $w, $count);
    $total += $count;
}

echo "  ─────────────────────\n";
printf("  %-12s %5d\n", 'TOTAL', $total);
echo "\nData in {$baseDir}\n";