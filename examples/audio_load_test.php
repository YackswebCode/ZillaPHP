<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Audio\Wav;
use ZillaPHP\Core\Application;

Application::boot();

$dir = __DIR__ . '/../data/audio/synthetic';
$files = glob("{$dir}/*.wav");

echo "Loading " . count($files) . " WAV files\n";
echo str_repeat('-', 72) . "\n";

foreach ($files as $path) {
    $wav = Wav::fromFile($path);
    $tensor = $wav->toTensor();

    printf("%-22s  %s\n",
        basename($path),
        $wav->summary()
    );
    printf("%-22s  tensor shape: %s  bytes: %s\n",
        '',
        $tensor->shape(),
        number_format(strlen(json_encode($tensor->data())))
    );

    // Print min/max to verify normalization
    $data = $tensor->data();
    printf("%-22s  min=%.4f  max=%.4f  mean=%.4f\n\n",
        '',
        min($data),
        max($data),
        array_sum($data) / count($data)
    );
}