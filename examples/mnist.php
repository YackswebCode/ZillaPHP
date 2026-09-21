<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\MnistDataset;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Training\Metrics\Accuracy;
use ZillaPHP\Training\Trainer;

Application::boot();

$base = __DIR__ . '/../data/mnist';

$train = new MnistDataset(
    "{$base}/train-images-idx3-ubyte",
    "{$base}/train-labels-idx1-ubyte",
);
$test = new MnistDataset(
    "{$base}/t10k-images-idx3-ubyte",
    "{$base}/t10k-labels-idx1-ubyte",
);

echo "Train: " . $train->size() . " samples\n";
echo "Test : " . $test->size() . " samples\n\n";

// ---- Subset for speed on CPU. Bump this later. ----
$TRAIN_N = 2000;
$TEST_N  = 500;
$EPOCHS  = 3;

$trainSamples = [];
for ($i = 0; $i < $TRAIN_N; $i++) $trainSamples[] = $train->get($i);

$testSamples = [];
for ($i = 0; $i < $TEST_N; $i++) $testSamples[] = $test->get($i);

$model = new Sequential([
    new Linear(784, 128),
    new ReLU(),
    new Linear(128, 10),
]);

$optimizer = new Adam($model->parameters(), lr: 0.001);
$loss      = new CrossEntropy();
$trainer   = new Trainer($model, $optimizer, $loss);

echo "Training MNIST subset ({$TRAIN_N} samples, {$EPOCHS} epochs)\n";
echo str_repeat('-', 60) . "\n";

$trainer->fit($trainSamples, epochs: $EPOCHS, onEpoch: function (int $e, float $l) {
    printf("Epoch %d   train loss = %.6f\n", $e, $l);
});

echo str_repeat('-', 60) . "\n";

// ---- Evaluate on held-out test set ----
$acc = new Accuracy();
foreach ($testSamples as [$x, $y]) {
    $acc->update($model->forward($x), $y);
}
printf("Test accuracy: %.2f%% (%d / %d)\n",
    100 * $acc->value(), $acc->correct(), $acc->total());

// ---- Save checkpoint ----
$serializer = new ModelSerializer();
$serializer->save($model, __DIR__ . '/../checkpoints/mnist.zilla.json', [
    'train_samples' => $TRAIN_N,
    'epochs'        => $EPOCHS,
    'test_accuracy' => $acc->value(),
]);
echo "Saved checkpoint: checkpoints/mnist.zilla.json\n";