<?php

declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Serialization\SafeTensorsSerializer;

Application::boot(native: true);

$model = new Sequential([
    new Linear(8, 4),
    new ReLU(),
    new Linear(4, 2),
]);

$path = __DIR__ . '/../checkpoints/demo.safetensors';

$serializer = new ModelSerializer(new SafeTensorsSerializer());
$serializer->save($model, $path, [
    'framework' => 'ZillaPHP',
    'version'   => '0.1.0-dev',
    'demo'      => true,
]);

echo "Saved: {$path}\n";
echo "Size:  " . filesize($path) . " bytes\n\n";

// ---- Load into a fresh model ----
$model2 = new Sequential([
    new Linear(8, 4),
    new ReLU(),
    new Linear(4, 2),
]);

$meta = $serializer->load($model2, $path);
echo "Metadata: " . json_encode($meta) . "\n\n";

// ---- Verify round-trip is bit-exact in FLOAT32 ----
$origParams   = $model->parameters();
$loadedParams = $model2->parameters();

$allMatch = true;
foreach ($origParams as $i => $p) {
    $orig = $p->data();
    $back = $loadedParams[$i]->data();

    $maxDiff = 0.0;
    foreach ($orig as $j => $v) {
        // What F32 storage "expects" the value to be
        $expected = unpack('f', pack('f', $v))[1];
        $maxDiff = max($maxDiff, abs($expected - $back[$j]));
    }

    $ok = $maxDiff < 1e-9;   // effectively bit-exact
    printf("param_%-2d  shape=%-10s  max_diff=%.3e  %s\n",
        $i, (string)$p->shape(), $maxDiff, $ok ? "OK" : "FAIL");

    $allMatch = $allMatch && $ok;
}

echo "\n";
echo $allMatch
    ? "Round-trip is bit-exact in F32.\n"
    : "ROUND-TRIP FAILED.\n";