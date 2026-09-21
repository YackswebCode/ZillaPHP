<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Training\Trainer;

Application::boot();

// ------- XOR dataset: 4 samples, 2 classes -------
//   x1  x2  →  class
//   0   0   →  0
//   0   1   →  1
//   1   0   →  1
//   1   1   →  0
$samples = [
    [Tensor::fromArray([[0.0, 0.0]]), [0]],
    [Tensor::fromArray([[0.0, 1.0]]), [1]],
    [Tensor::fromArray([[1.0, 0.0]]), [1]],
    [Tensor::fromArray([[1.0, 1.0]]), [0]],
];

// Hidden layer makes the network non-linear.
$model = new Sequential([
    new Linear(2, 8),
    new ReLU(),
    new Linear(8, 2),
]);

$optimizer = new Adam($model->parameters(), lr: 0.05);
$loss      = new CrossEntropy();
$trainer   = new Trainer($model, $optimizer, $loss);

echo "Training XOR (Linear(2,8) → ReLU → Linear(8,2))\n";
echo str_repeat('-', 55) . "\n";

$trainer->fit($samples, epochs: 500, onEpoch: function (int $e, float $loss) {
    if ($e % 50 === 0 || $e === 499) {
        printf("Epoch %3d  loss = %.8f\n", $e, $loss);
    }
});

echo str_repeat('-', 55) . "\n";
echo "Predictions:\n";

foreach ($samples as [$x, $target]) {
    $logits = $model->forward($x);
    $probs  = $logits->softmax()->data();
    $pred   = $probs[0] > $probs[1] ? 0 : 1;
    printf("  [%.0f, %.0f] → predicted %d  (expected %d)  probs=[%.4f, %.4f]\n",
        $x->data()[0], $x->data()[1],
        $pred, $target[0], $probs[0], $probs[1]);
}