<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

$hasCuda = is_file(__DIR__ . '/../native/cuda/libzilla_cuda.so');

echo "CUDA library present: " . ($hasCuda ? 'yes' : 'no') . "\n";

$sizes = [
    [128, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [1024, 1024, 1024],
];

// Pre-generate test data so generation time doesn't pollute timings
$matrices = [];
foreach ($sizes as $i => [$m, $k, $n]) {
    $matrices[$i] = [
        'a' => Tensor::randn([$m, $k]),
        'b' => Tensor::randn([$k, $n]),
    ];
}

function bench(Tensor $a, Tensor $b, int $reps = 5): float
{
    // Warmup — this is where context init and first alloc happen
    $a->matmul($b);
    $a->matmul($b);

    $best = INF;
    for ($r = 0; $r < $reps; $r++) {
        $t0 = microtime(true);
        $a->matmul($b);
        $dt = microtime(true) - $t0;
        if ($dt < $best) $best = $dt;
    }
    return $best * 1000;
}

// ==================================================================
// CPU
// ==================================================================
Application::boot(native: true, cuda: false);
printf("CPU backend: %s\n\n", Tensor::backend()->name());

$cpuTimes = [];
foreach ($sizes as $i => [$m, $k, $n]) {
    $cpuTimes[$i] = bench($matrices[$i]['a'], $matrices[$i]['b']);
}

// ==================================================================
// CUDA
// ==================================================================
$cudaTimes = [];
if ($hasCuda) {
    Application::boot(cuda: true, native: true);
    printf("CUDA backend: %s\n\n", Tensor::backend()->name());

    if (Tensor::backend()->name() === 'cuda') {
        foreach ($sizes as $i => [$m, $k, $n]) {
            $cudaTimes[$i] = bench($matrices[$i]['a'], $matrices[$i]['b']);
        }
    }
}

// ==================================================================
// Report
// ==================================================================
printf("%-16s %-16s %-16s %-16s %-12s\n",
    'A', 'B', 'CPU+native', 'CUDA', 'GPU speedup');
echo str_repeat('-', 82) . "\n";

foreach ($sizes as $i => [$m, $k, $n]) {
    $cpu = $cpuTimes[$i];
    $cuda = $cudaTimes[$i] ?? null;

    $speedup = '';
    if ($cuda !== null && $cuda > 0) {
        $speedup = sprintf('%.2fx', $cpu / $cuda);
    }

    printf("%-16s %-16s %-16s %-16s %-12s\n",
        "{$m}x{$k}",
        "{$k}x{$n}",
        sprintf('%.2f ms', $cpu),
        $cuda === null ? 'N/A' : sprintf('%.2f ms', $cuda),
        $speedup,
    );
}