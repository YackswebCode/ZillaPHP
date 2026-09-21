<?php

declare(strict_types=1);

namespace ZillaPHP\Loss;

use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

final class CrossEntropy extends Loss
{
    /**
     * Fused numerically-stable softmax + NLL in a single backward pass.
     *
     * @param Tensor        $prediction  logits [batch, classes]
     * @param Tensor|int[]  $target      class indices [batch]
     */
    public function forward(Tensor $prediction, Tensor|array $target): Tensor
    {
        $targets = $target instanceof Tensor
            ? array_map('intval', $target->data())
            : array_map('intval', $target);

        $dims = $prediction->shape()->dims();
        if (count($dims) !== 2) {
            throw new \RuntimeException("CrossEntropy expects 2D logits.");
        }
        [$batch, $classes] = $dims;

        if (count($targets) !== $batch) {
            throw new \RuntimeException(
                "Target count ({$batch}) does not match batch size ({$batch})."
            );
        }

        // ---- Forward: numerically-stable softmax + NLL ----
        $logits = $prediction->data();
        $probs  = [];

        for ($i = 0; $i < $batch; $i++) {
            $row = array_slice($logits, $i * $classes, $classes);
            $max = max($row);
            $sum = 0.0;
            $exps = [];
            foreach ($row as $v) {
                $e = exp($v - $max);
                $exps[] = $e;
                $sum += $e;
            }
            foreach ($exps as $e) {
                $probs[] = $e / $sum;
            }
        }

        $loss = 0.0;
        for ($i = 0; $i < $batch; $i++) {
            $p = $probs[$i * $classes + $targets[$i]];
            $loss += -log(max($p, 1e-12));
        }
        $loss /= $batch;

        $result = new Tensor([$loss], new Shape([1]));

        if ($prediction->requiresGrad()) {
            Tensor::attachAutograd(
                $result,
                [$prediction],
                'cross_entropy',
                function (Tensor $g) use ($prediction, $probs, $targets, $batch, $classes): void {
                    $scale = (float) $g->item() / $batch;
                    $dx = [];
                    for ($i = 0; $i < $batch; $i++) {
                        for ($j = 0; $j < $classes; $j++) {
                            $p  = $probs[$i * $classes + $j];
                            $oh = ($j === $targets[$i]) ? 1.0 : 0.0;
                            $dx[] = ($p - $oh) * $scale;
                        }
                    }
                    $prediction->accumulateGradPublic(new Tensor($dx, $prediction->shape()));
                },
            );
        }

        return $result;
    }
}