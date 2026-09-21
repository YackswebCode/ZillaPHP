<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\ArrayDataset;
use ZillaPHP\Data\DataLoader;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Training\Metrics\Accuracy;
use ZillaPHP\Training\Trainer;

Application::boot();

// ---- Generate 10 clusters arranged on a circle ----
$classes = 10;
$perClass = 200;

$items = [];
for ($c = 0; $c < $classes; $c++) {
    $angle = 2 * M_PI * $c / $classes;
    $cx = 3.0 * cos($angle);
    $cy = 3.0 * sin($angle);

    for ($i = 0; $i < $perClass; $i++) {
        $x1 = $cx + gauss() * 0.35;
        $x2 = $cy + gauss() * 0.35;
        $items[] = [
            Tensor::fromArray([[$x1, $x2]]),   // input [1, 2]
            [$c],                              // target [1]
        ];
    }
}
shuffle($items);

$dataset = new ArrayDataset($items);
$loader  = new DataLoader($dataset, batchSize: 64, shuffle: true);

$model = new Sequential([
    new Linear(2, 32),
    new ReLU(),
    new Linear(32, 32),
    new ReLU(),
    new Linear(32, $classes),
]);

$optimizer = new Adam($model->parameters(), lr: 0.01);
$loss      = new CrossEntropy();
$acc       = new Accuracy();

echo "Training 10-class classifier on synthetic 2D Gaussians\n";
echo str_repeat('-', 60) . "\n";

for ($epoch = 0; $epoch < 30; $epoch++) {
    $acc->reset();
    $totalLoss = 0.0; $batches = 0;

    foreach ($loader as $batch) {
        // For simplicity, train one sample at a time on small batches.
        // A true batched matmul is a later optimization.
        foreach ($batch as [$x, $y]) {
            $optimizer->zeroGrad();
            $logits = $model->forward($x);
            $l      = $loss->forward($logits, $y);
            $l->backward();
            $optimizer->step();

            $totalLoss += (float) $l->item();
            $acc->update($logits, $y);
            $batches++;
        }
    }

    printf("Epoch %2d   avg loss = %.6f   accuracy = %.2f%%\n",
        $epoch, $totalLoss / max($batches, 1), 100 * $acc->value());
}

echo str_repeat('-', 60) . "\n";
echo "Final accuracy: " . number_format(100 * $acc->value(), 2) . "%\n";

function gauss(): float
{
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    return sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2);
}