<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\MnistDataset;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Conv2D;
use ZillaPHP\NN\Layers\Flatten;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Layers\MaxPool2D;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Tensor\Shape\Shape;  
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Training\Metrics\Accuracy;
use ZillaPHP\Training\Trainer;

Application::boot(native: true);

$TRAIN_N = (int) (getenv('TRAIN_N') ?: 60000);
$TEST_N  = (int) (getenv('TEST_N')  ?: 10000);
$EPOCHS  = (int) (getenv('EPOCHS')  ?: 3);
$LR      = (float) (getenv('LR')    ?: 0.001);

$base = __DIR__ . '/../data/mnist';
$train = new MnistDataset("{$base}/train-images-idx3-ubyte", "{$base}/train-labels-idx1-ubyte");
$test  = new MnistDataset("{$base}/t10k-images-idx3-ubyte", "{$base}/t10k-labels-idx1-ubyte");

echo "ZillaPHP — MNIST CNN\n";
echo str_repeat('=', 60) . "\n";
printf("Train: %d   Test: %d   Epochs: %d   LR: %.4f\n",
    $TRAIN_N, $TEST_N, $EPOCHS, $LR);
printf("Backend: %s\n", Tensor::backend()->name());
echo str_repeat('=', 60) . "\n\n";

// ---- Convert MNIST [1, 784] into [1, 28, 28] per-sample ----
function mnistToCHW(Tensor $flat): Tensor
{
    return new Tensor($flat->data(), new Shape([1, 28, 28]));
}

// ---- Model ----
$model = new Sequential([
    new Conv2D(1, 8, kernelSize: 3, stride: 1, padding: 1),
    new ReLU(),
    new MaxPool2D(2),
    new Conv2D(8, 16, kernelSize: 3, stride: 1, padding: 1),
    new ReLU(),
    new MaxPool2D(2),
    new Flatten(),
    new Linear(16 * 7 * 7, 10),
]);

$totalParams = array_sum(array_map(fn($p) => $p->shape()->size(), $model->parameters()));
printf("Params: %d\n\n", $totalParams);

$optimizer = new Adam($model->parameters(), lr: $LR);
$loss      = new CrossEntropy();

// ---- Train per-sample (pure PHP conv is too slow for batching) ----
$startAll = microtime(true);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $start = microtime(true);
    $totalLoss = 0.0;

    for ($i = 0; $i < $TRAIN_N; $i++) {
        [$xFlat, $y] = $train->get($i);
        $x = mnistToCHW($xFlat);

        $optimizer->zeroGrad();
        $pred = $model->forward($x);            // [1, 10]
        $l    = $loss->forward($pred, $y);      // scalar
        $l->backward();
        $optimizer->step();

        $totalLoss += (float) $l->item();

        if (($i + 1) % 100 === 0) {
            $elapsed = microtime(true) - $start;
            printf("  epoch %d  %4d/%d  loss=%.4f  %.1fs\n",
                $epoch, $i + 1, $TRAIN_N,
                $totalLoss / ($i + 1), $elapsed);
        }
    }

    // ---- Evaluate ----
    $acc = new Accuracy();
    for ($i = 0; $i < $TEST_N; $i++) {
        [$xFlat, $y] = $test->get($i);
        $x = mnistToCHW($xFlat);
        $logits = $model->forward($x);
        $acc->update($logits, $y);
    }

    $elapsed = microtime(true) - $start;
    printf("Epoch %d  avg-loss=%.4f  test=%.2f%% (%d/%d)  time=%.1fs\n\n",
        $epoch,
        $totalLoss / $TRAIN_N,
        100 * $acc->value(),
        $acc->correct(),
        $acc->total(),
        $elapsed,
    );
}

$total = microtime(true) - $startAll;
printf("Total: %.1f s (%.1f min)\n", $total, $total / 60);