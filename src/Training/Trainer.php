<?php

declare(strict_types=1);

namespace ZillaPHP\Training;

use ZillaPHP\Data\DataLoader;
use ZillaPHP\Loss\Loss;
use ZillaPHP\NN\Module;
use ZillaPHP\Optim\Optimizer;
use ZillaPHP\Tensor\Tensor;

final class Trainer
{
    public function __construct(
        private Module $model,
        private Optimizer $optimizer,
        private Loss $loss,
    ) {}

    /**
     * Per-sample training.
     *
     * Simple, and correct, but slow — every sample triggers a full
     * forward + backward + optimizer step. Use fitLoader() for real
     * workloads.
     *
     * @param array<array{0: Tensor|array, 1: Tensor|array}> $data
     * @return float[]
     */
    public function fit(array $data, int $epochs = 100, ?callable $onEpoch = null): array
    {
        $history = [];

        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            $totalLoss = 0.0;
            $count = 0;

            foreach ($data as [$input, $target]) {
                $x = $input instanceof Tensor ? $input : Tensor::fromArray($input);
                $y = $target;

                $this->optimizer->zeroGrad();

                $pred = $this->model->forward($x);
                $loss = $this->loss->forward($pred, $y);

                $loss->backward();
                $this->optimizer->step();

                $totalLoss += (float) $loss->item();
                $count++;
            }

            $avg = $totalLoss / max($count, 1);
            $history[] = $avg;

            if ($onEpoch !== null) {
                $onEpoch($epoch, $avg);
            }
        }
        return $history;
    }

    /**
     * Batched training driven by a DataLoader.
     *
     * Each batch is stacked into a single [N, D] input tensor so the
     * forward/backward pass runs once per batch, not once per sample.
     * This is where the real speedup comes from on CPU.
     *
     * @return float[]
     */
    public function fitLoader(
        DataLoader $loader,
        int $epochs = 10,
        ?callable $onEpoch = null,
    ): array {
        $history = [];

        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            $totalLoss = 0.0;
            $batches = 0;

            foreach ($loader as $batch) {
                $inputs  = [];
                $targets = [];

                foreach ($batch as $item) {
                    /** @var Tensor $t */
                    $t = $item[0];                 // [1, D]
                    $y = $item[1];                 // [int]

                    $inputs[]  = $t;
                    $targets[] = (int) (is_array($y) ? $y[0] : $y);
                }

                $x = Tensor::concatRows($inputs);   // [N, D]

                $this->optimizer->zeroGrad();

                $pred = $this->model->forward($x);              // [N, classes]
                $loss = $this->loss->forward($pred, $targets);  // scalar
                $loss->backward();
                $this->optimizer->step();

                $totalLoss += (float) $loss->item();
                $batches++;
            }

            $avg = $totalLoss / max($batches, 1);
            $history[] = $avg;

            if ($onEpoch !== null) {
                $onEpoch($epoch, $avg);
            }
        }

        return $history;
    }
}