<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\DataLoader;
use ZillaPHP\Data\MnistDataset;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Training\Metrics\Accuracy;
use ZillaPHP\Training\Trainer;

Application::boot(native: true);

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

// ---- Config ----
$TRAIN_N  = (int) ($_ENV['TRAIN_N']  ?? 6000);   // subset for speed
$TEST_N   = (int) ($_ENV['TEST_N']   ?? 1000);
$BATCH    = (int) ($_ENV['BATCH']    ?? 32);
$EPOCHS   = (int) ($_ENV['EPOCHS']   ?? 5);
$LR       = (float) ($_ENV['LR']     ?? 0.001);

// ---- Slice dataset ----
$trainItems = [];
for ($i = 0; $i < $TRAIN_N; $i++) $trainItems[] = $train->get($i);

$testItems = [];
for ($i = 0; $i < $TEST_N; $i++) $testItems[] = $test->get($i);

$trainLoader = new DataLoader(new \ZillaPHP\Data\ArrayDataset($trainItems), $BATCH, shuffle: true);

// ---- Model ----
$model = new Sequential([
    new Linear(784, 128),
    new ReLU(),
    new Linear(128, 10),
]);

$optimizer = new Adam($model->parameters(), lr: $LR);
$loss      = new CrossEntropy();
$trainer   = new Trainer($model, $optimizer, $loss);

echo "Batched training — {$TRAIN_N} samples, batch={$BATCH}, epochs={$EPOCHS}\n";
echo str_repeat('-', 60) . "\n";

$start = microtime(true);

$trainer->fitLoader($trainLoader, epochs: $EPOCHS, onEpoch: function (int $e, float $l) {
    printf("Epoch %d   avg loss = %.6f\n", $e, $l);
});

$elapsed = microtime(true) - $start;
printf("Training time: %.2f s\n", $elapsed);

echo str_repeat('-', 60) . "\n";

// ---- Evaluate ----
$acc = new Accuracy();
$evalBatch = 64;
for ($i = 0; $i < count($testItems); $i += $evalBatch) {
    $chunk = array_slice($testItems, $i, $evalBatch);
    $inputs = array_map(fn($x) => $x[0], $chunk);
    $targets = array_map(fn($x) => $x[1][0], $chunk);

    $x = \ZillaPHP\Tensor\Tensor::concatRows($inputs);
    $logits = $model->forward($x);
    $acc->update($logits, $targets);
}

printf("Test accuracy: %.2f%% (%d / %d)\n",
    100 * $acc->value(), $acc->correct(), $acc->total());

// ---- Save ----
$serializer = new ModelSerializer();
$serializer->save($model, __DIR__ . '/../checkpoints/mnist_batched.zilla.json', [
    'train_samples' => $TRAIN_N,
    'batch_size'    => $BATCH,
    'epochs'        => $EPOCHS,
    'test_accuracy' => $acc->value(),
]);
echo "Saved checkpoint: checkpoints/mnist_batched.zilla.json\n";