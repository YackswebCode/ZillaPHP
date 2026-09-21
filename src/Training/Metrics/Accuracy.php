<?php

declare(strict_types=1);

namespace ZillaPHP\Training\Metrics;

use ZillaPHP\Tensor\Tensor;

final class Accuracy
{
    private int $correct = 0;
    private int $total   = 0;

    /**
     * @param Tensor   $logits   [batch, classes]
     * @param int[]    $targets  [batch]
     */
    public function update(Tensor $logits, array $targets): void
    {
        $dims = $logits->shape()->dims();
        [$batch, $classes] = $dims;
        $d = $logits->data();

        for ($i = 0; $i < $batch; $i++) {
            $best = -INF; $bestIdx = 0;
            for ($j = 0; $j < $classes; $j++) {
                $v = $d[$i * $classes + $j];
                if ($v > $best) { $best = $v; $bestIdx = $j; }
            }
            if ($bestIdx === $targets[$i]) $this->correct++;
            $this->total++;
        }
    }

    public function value(): float
    {
        return $this->total === 0 ? 0.0 : $this->correct / $this->total;
    }

    public function reset(): void
    {
        $this->correct = 0;
        $this->total   = 0;
    }

    public function correct(): int { return $this->correct; }
    public function total(): int   { return $this->total; }
}