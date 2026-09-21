<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

Application::boot();

echo "=== Example 1: y = x² at x = 2, expect dy/dx = 4 ===\n";
$x = Tensor::fromArray([2.0], requiresGrad: true);
$y = $x->mul($x);
$y->backward();
echo "dy/dx = " . $x->grad()->item() . "\n\n";

echo "=== Example 2: chain rule, y = (x+1)² at x = 1, expect dy/dx = 4 ===\n";
$x = Tensor::fromArray([1.0], requiresGrad: true);
$one = Tensor::fromArray([1.0]);
$y = $x->add($one)->mul($x->add($one));
$y->backward();
echo "dy/dx = " . $x->grad()->item() . "\n\n";

echo "=== Example 3: y = sum(A @ B) ===\n";
$A = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]], requiresGrad: true);
$B = Tensor::fromArray([[5.0, 6.0], [7.0, 8.0]], requiresGrad: true);
$C = $A->matmul($B);
$loss = $C->sum();
$loss->backward();

echo "dL/dA = " . json_encode($A->grad()->data()) . "\n";
echo "dL/dB = " . json_encode($B->grad()->data()) . "\n";