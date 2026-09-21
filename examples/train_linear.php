<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Loss\MSE;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Training\Trainer;

Application::boot();

// ---- 1. Ground-truth: y = 2x + 1 ----
$wTrue = 2.0;
$bTrue = 1.0;

$samples = [];
for ($i = 0; $i < 256; $i++) {
    $x = (mt_rand() / mt_getrandmax()) * 4.0 - 2.0;   // uniform in [-2, 2]
    $y = $wTrue * $x + $bTrue;
    $samples[] = [
        Tensor::fromArray([[$x]]),   // [1, 1]
        Tensor::fromArray([[$y]]),   // [1, 1]
    ];
}

// ---- 2. Model: Linear(1, 1) ----
$model = new Sequential([
    new Linear(1, 1),
]);

$optimizer = new Adam($model->parameters(), lr: 0.05);
$loss      = new MSE();

$trainer = new Trainer($model, $optimizer, $loss);

echo "Training Linear(1,1) to fit y = 2x + 1\n";
echo str_repeat('-', 50) . "\n";

$trainer->fit($samples, epochs: 200, onEpoch: function (int $epoch, float $loss) {
    if ($epoch % 20 === 0 || $epoch === 199) {
        printf("Epoch %3d  loss = %.8f\n", $epoch, $loss);
    }
});

echo str_repeat('-', 50) . "\n";
$lin = $model->layers()[0];
printf("Learned weight : %.5f   (true: 2.0)\n", $lin->weight()->data()[0]);
printf("Learned bias   : %.5f   (true: 1.0)\n", $lin->bias()->data()[0]);