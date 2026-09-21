<?php

declare(strict_types=1);

namespace ZillaPHP\Generators;

use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Tokenizers\CharTokenizer;
use ZillaPHP\Transformers\TransformerLM;

/**
 * Autoregressive text sampler with KV cache.
 *
 * First step processes the full prompt and populates the cache.
 * Each subsequent step processes only the new token (1 forward step
 * on a [1, dim] input) using cached K/V from all previous positions.
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

        if (!$this->model instanceof TransformerLM) {
            throw new \RuntimeException(
                "Sampler requires a TransformerLM (with cache support)."
            );
        }

        return Tensor::noGrad(function () use ($prompt, $maxNewTokens, $temperature) {
            $ids   = $this->tokenizer->encode($prompt);
            $vocab = $this->tokenizer->vocabSize();

            // Truncate prompt to maxContext
            if (count($ids) > $this->maxContext) {
                $ids = array_slice($ids, -$this->maxContext);
            }

            // ---- Initial pass: full prompt ----
            $cache = $this->model->newCache();
            $input = Tensor::fromArray(array_map('floatval', $ids));
            $logits = $this->model->forwardWithCache($input, $cache);

            // ---- Generate tokens one at a time ----
            for ($step = 0; $step < $maxNewTokens; $step++) {
                $T = $logits->shape()->dims()[0];
                $all = $logits->data();
                $lastRow = array_slice($all, ($T - 1) * $vocab, $vocab);

                $next = $this->sampleFromLogits($lastRow, $temperature);
                $ids[] = $next;

                // ---- Stop if cache is full ----
                if ($cache->length >= $this->maxContext) {
                    // Sliding window: reset cache, re-encode last maxContext tokens
                    $ids = array_slice($ids, -$this->maxContext);
                    $cache = $this->model->newCache();
                    $input = Tensor::fromArray(array_map('floatval', $ids));
                    $logits = $this->model->forwardWithCache($input, $cache);
                    continue;
                }

                // ---- Forward single new token ----
                $input = Tensor::fromArray([(float) $next]);
                $logits = $this->model->forwardWithCache($input, $cache);
            }

            return $this->tokenizer->decode($ids);
        });
    }

    /** @param float[] $logits */
    private function sampleFromLogits(array $logits, float $temperature): int
    {
        if ($temperature > 0) {
            foreach ($logits as $i => $v) {
                $logits[$i] = $v / $temperature;
            }
        }

        $max = max($logits);
        $exps = [];
        $sum = 0.0;
        foreach ($logits as $v) {
            $e = exp($v - $max);
            $exps[] = $e;
            $sum += $e;
        }
        foreach ($exps as $i => $e) $exps[$i] = $e / $sum;

        $r = mt_rand() / mt_getrandmax();
        $acc = 0.0;
        foreach ($exps as $i => $p) {
            $acc += $p;
            if ($r < $acc) return $i;
        }
        return count($exps) - 1;
    }
}