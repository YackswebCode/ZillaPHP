<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

Application::boot();

$a = Tensor::fromArray([
    [1.0, 2.0],
    [3.0, 4.0],
]);

$b = Tensor::fromArray([
    [5.0, 6.0],
    [7.0, 8.0],
]);

echo "A = {$a}\n";
echo "B = {$b}\n\n";

$c = $a->add($b);
echo "A + B = " . json_encode($c->data()) . "\n";

$m = $a->matmul($b);
echo "A @ B = " . json_encode($m->data()) . "\n";

$r = $a->relu();
echo "ReLU(A) = " . json_encode($r->data()) . "\n";

$s = $a->sum();
echo "Sum(A) = " . $s->item() . "\n";