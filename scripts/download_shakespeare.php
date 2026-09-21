<?php

declare(strict_types=1);

/**
 * Downloads the Tiny Shakespeare corpus (~1.1 MB) into data/shakespeare.txt.
 */

$target = __DIR__ . '/../data/shakespeare.txt';
if (is_file($target)) {
    echo "Already present: {$target}\n";
    exit(0);
}

$url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';

echo "Downloading Tiny Shakespeare...\n";
$data = @file_get_contents($url);
if ($data === false) {
    fwrite(STDERR, "Failed to download {$url}\n");
    exit(1);
}

if (!is_dir(dirname($target))) mkdir(dirname($target), 0777, true);
file_put_contents($target, $data);

$chars = strlen($data);
$unique = count(array_unique(str_split($data)));
printf("Saved %d bytes, %d unique characters to %s\n",
    $chars, $unique, $target);