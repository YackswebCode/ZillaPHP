<?php
require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

$A = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
$B = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]];

Application::boot(native: true);
$cpuC = Tensor::fromArray($A)->matmul(Tensor::fromArray($B));
echo "CPU backend: " . Tensor::backend()->name() . "\n";
echo "CPU matmul:  " . json_encode($cpuC->data()) . "\n";

Application::boot(cuda: true);
$gpuC = Tensor::fromArray($A)->matmul(Tensor::fromArray($B));
echo "GPU backend: " . Tensor::backend()->name() . "\n";
echo "GPU matmul:  " . json_encode($gpuC->data()) . "\n";

$maxErr = 0.0;
foreach ($cpuC->data() as $i => $v) {
    $maxErr = max($maxErr, abs($v - $gpuC->data()[$i]));
}
printf("Max error: %.6f\n", $maxErr);
echo ($maxErr < 1e-3) ? "PARITY OK\n" : "PARITY FAILED\n";