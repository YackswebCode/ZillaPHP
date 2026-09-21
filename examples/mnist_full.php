<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Data\DataLoader;
use ZillaPHP\Data\Dataset;
use ZillaPHP\Data\MnistDataset;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Training\Metrics\Accuracy;

Application::boot(native: true);

// ====================================================================
// Configuration — reads real environment variables
// ====================================================================
$TRAIN_N    = (int)   (getenv('TRAIN_N')    ?: 60000);
$TEST_N     = (int)   (getenv('TEST_N')     ?: 10000);
$BATCH      = (int)   (getenv('BATCH')      ?: 64);
$EPOCHS     = (int)   (getenv('EPOCHS')     ?: 15);
$LR         = (float) (getenv('LR')         ?: 0.001);
$PATIENCE   = (int)   (getenv('PATIENCE')   ?: 3);
$CHECKPOINT = getenv('CHECKPOINT') ?: __DIR__ . '/../checkpoints/mnist_best.zilla.json';

$base = __DIR__ . '/../data/mnist';

$train = new MnistDataset(
    "{$base}/train-images-idx3-ubyte",
    "{$base}/train-labels-idx1-ubyte",
);
$test = new MnistDataset(
    "{$base}/t10k-images-idx3-ubyte",
    "{$base}/t10k-labels-idx1-ubyte",
);

echo "ZillaPHP — Full MNIST Training\n";
echo str_repeat('=', 66) . "\n";
printf("Train: %d samples   Test: %d samples\n", $TRAIN_N, $TEST_N);
printf("Batch: %d   Epochs: %d   LR: %.4f   Patience: %d\n",
    $BATCH, $EPOCHS, $LR, $PATIENCE);
printf("Backend: %s\n", Tensor::backend()->name());
echo str_repeat('=', 66) . "\n\n";

// ---- Lazy loader — no materialization of the whole dataset --------
$trainLoader = new DataLoader($train, $BATCH, shuffle: true);

// ---- Model: 784 → 256 → 128 → 10 ----------------------------------
$model = new Sequential([
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 128),
    new ReLU(),
    new Linear(128, 10),
]);

$optimizer = new Adam($model->parameters(), lr: $LR);
$loss      = new CrossEntropy();

// ---- Lazy batched evaluation --------------------------------------
function evaluateModel(
    Sequential $model,
    Dataset $test,
    int $testN,
    int $batch = 128,
): array {
    $acc = new Accuracy();

    for ($i = 0; $i < $testN; $i += $batch) {
        $end     = min($i + $batch, $testN);
        $inputs  = [];
        $targets = [];

        for ($j = $i; $j < $end; $j++) {
            [$x, $y] = $test->get($j);
            $inputs[]  = $x;
            $targets[] = (int) $y[0];
        }

        $x      = Tensor::concatRows($inputs);
        $logits = $model->forward($x);
        $acc->update($logits, $targets);
    }

    return [
        'accuracy' => $acc->value(),
        'correct'  => $acc->correct(),
        'total'    => $acc->total(),
    ];
}

// ---- Training loop with best-checkpoint tracking -------------------
$serializer = new ModelSerializer();
$bestAcc    = 0.0;
$bestEpoch  = -1;
$noImprove  = 0;
$startAll   = microtime(true);
$processed  = 0;
$epochTotal = $EPOCHS;

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    $startEpoch = microtime(true);

    // ---- Train one epoch (only over the first $TRAIN_N samples) ----
    $totalLoss = 0.0;
    $batches   = 0;

    foreach ($trainLoader as $batch) {
        // stop once we've consumed TRAIN_N samples this epoch
        if ($processed >= $TRAIN_N) {
            $processed = 0;
            break;
        }

        $inputs  = [];
        $targets = [];
        $used = 0;
        foreach ($batch as $item) {
            if ($processed >= $TRAIN_N) break;
            $inputs[]  = $item[0];
            $targets[] = (int) $item[1][0];
            $processed++;
            $used++;
        }
        if ($used === 0) continue;

        $x = Tensor::concatRows($inputs);

        $optimizer->zeroGrad();
        $pred = $model->forward($x);
        $l    = $loss->forward($pred, $targets);
        $l->backward();
        $optimizer->step();

        $totalLoss += (float) $l->item();
        $batches++;
    }
    // Reset counter for next epoch
    $processed = 0;

    $avgLoss = $totalLoss / max($batches, 1);

    // ---- Evaluate on test set (lazy) ----
    $metrics = evaluateModel($model, $test, $TEST_N);
    $testAcc = $metrics['accuracy'];

    $elapsed = microtime(true) - $startEpoch;

    printf("Epoch %2d  loss=%.5f  test=%.2f%% (%d/%d)  time=%.1fs\n",
        $epoch,
        $avgLoss,
        100 * $testAcc,
        $metrics['correct'],
        $metrics['total'],
        $elapsed,
    );

    // ---- Best-checkpoint tracking ----
    if ($testAcc > $bestAcc) {
        $bestAcc   = $testAcc;
        $bestEpoch = $epoch;
        $noImprove = 0;

        $serializer->save($model, $CHECKPOINT, [
            'epoch'         => $epoch,
            'loss'          => $avgLoss,
            'test_accuracy' => $testAcc,
            'train_samples' => $TRAIN_N,
        ]);
        echo "           ↑ new best — saved to " . basename($CHECKPOINT) . "\n";
    } else {
        $noImprove++;
        if ($noImprove >= $PATIENCE) {
            echo "           early stop (no improvement for {$PATIENCE} epochs)\n";
            break;
        }
    }
}

$totalTime = microtime(true) - $startAll;

echo str_repeat('=', 66) . "\n";
printf("Best test accuracy: %.2f%% at epoch %d\n", 100 * $bestAcc, $bestEpoch);
printf("Total training time: %.1f s (%.1f min)\n",
    $totalTime, $totalTime / 60);
echo "Best checkpoint: {$CHECKPOINT}\n";