<?php

declare(strict_types=1);

namespace ZillaPHP\Generators;

use ZillaPHP\NN\Module;
use ZillaPHP\Tokenizers\CharTokenizer;
use ZillaPHP\Tensor\Tensor;

/**
 * Autoregressive text sampler.
 */
final class Sampler
{
    public function __construct(
        private Module $model,
        private CharTokenizer $tokenizer,
        private int $maxContext = 64,
    ) {}

    public function generate(
        string $prompt,
        int $maxNewTokens = 200,
        float $temperature = 0.8,
        int $seed = 0,
    ): string {
        if ($seed !== 0) mt_srand($seed);

        $ids = $this->tokenizer->encode($prompt);
        $vocab = $this->tokenizer->vocabSize();

        for ($step = 0; $step < $maxNewTokens; $step++) {
            $context = count($ids) > $this->maxContext
                ? array_slice($ids, -$this->maxContext)
                : $ids;

            $input = Tensor::fromArray(array_map('floatval', $context));
            $logits = $this->model->forward($input);

            $T = count($context);
            $all = $logits->data();
            $lastLogits = array_slice($all, ($T - 1) * $vocab, $vocab);

            if ($temperature > 0) {
                foreach ($lastLogits as $i => $v) {
                    $lastLogits[$i] = $v / $temperature;
                }
            }

            $max = max($lastLogits);
            $exps = [];
            $sum = 0.0;
            foreach ($lastLogits as $v) {
                $e = exp($v - $max);
                $exps[] = $e;
                $sum += $e;
            }
            foreach ($exps as $i => $e) $exps[$i] = $e / $sum;

            $next = $this->sampleFromDistribution($exps);
            $ids[] = $next;
        }

        return $this->tokenizer->decode($ids);
    }

    /** @param float[] $probs */
    private function sampleFromDistribution(array $probs): int
    {
        $r = mt_rand() / mt_getrandmax();
        $acc = 0.0;
        foreach ($probs as $i => $p) {
            $acc += $p;
            if ($r < $acc) return $i;
        }
        return count($probs) - 1;
    }
}