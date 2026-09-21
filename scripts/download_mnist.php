<?php

declare(strict_types=1);

/**
 * Downloads and extracts the original MNIST dataset into ./data/mnist/.
 * Uses a public mirror that does not require authentication.
 */

$baseDir = __DIR__ . '/../data/mnist';
if (!is_dir($baseDir)) mkdir($baseDir, 0777, true);

$files = [
    'train-images-idx3-ubyte.gz' => 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz' => 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz'  => 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'  => 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
];

foreach ($files as $name => $url) {
    $gzPath = "{$baseDir}/{$name}";
    $rawPath = "{$baseDir}/" . str_replace('.gz', '', $name);

    if (is_file($rawPath)) {
        echo "Already present: {$rawPath}\n";
        continue;
    }

    if (!is_file($gzPath)) {
        echo "Downloading {$name}...\n";
        $data = @file_get_contents($url);
        if ($data === false) {
            fwrite(STDERR, "Failed to download {$url}\n");
            exit(1);
        }
        file_put_contents($gzPath, $data);
    }

    echo "Extracting {$name}...\n";
    $raw = gzdecode(file_get_contents($gzPath));
    if ($raw === false) {
        fwrite(STDERR, "Failed to decompress {$gzPath}\n");
        exit(1);
    }
    file_put_contents($rawPath, $raw);
}

echo "MNIST ready in {$baseDir}\n";