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
 *
 * Two separate buffers are maintained:
 *
 *   $outputIds   — every token generated so far (for final decoding)
 *   $workingIds  — the sliding context window (feeds the model)
 *
 * Truncating $workingIds when the cache fills up must NOT lose the
 * tokens already emitted to the user.
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
            $vocab = $this->tokenizer->vocabSize();

            // ---- Prepare initial context ----
            $promptIds = $this->tokenizer->encode($prompt);

            // Truncate prompt if it exceeds the context window
            if (count($promptIds) > $this->maxContext) {
                $promptIds = array_slice($promptIds, -$this->maxContext);
            }

            // $outputIds accumulates EVERYTHING for the final decode.
            // $workingIds is the sliding window that feeds the model.
            $outputIds  = $promptIds;
            $workingIds = $promptIds;

            // ---- Initial forward pass over the prompt ----
            $cache  = $this->model->newCache();
            $input  = Tensor::fromArray(array_map('floatval', $workingIds));
            $logits = $this->model->forwardWithCache($input, $cache);

            // ---- Autoregressive loop ----
            for ($step = 0; $step < $maxNewTokens; $step++) {
                $T   = $logits->shape()->dims()[0];
                $all = $logits->data();
                $lastRow = array_slice($all, ($T - 1) * $vocab, $vocab);

                $next = $this->sampleFromLogits($lastRow, $temperature);

                // Append to BOTH buffers
                $outputIds[]  = $next;   // never truncated
                $workingIds[] = $next;   // may be truncated below

                // ---- Sliding window when the cache would overflow ----
                if ($cache->length + 1 > $this->maxContext) {
                    // Keep only half the window so we get many fast steps
                    // before the next reset.
                    $keep = max(1, intdiv($this->maxContext, 2));
                    $workingIds = array_slice($workingIds, -$keep);

                    $cache  = $this->model->newCache();
                    $input  = Tensor::fromArray(array_map('floatval', $workingIds));
                    $logits = $this->model->forwardWithCache($input, $cache);
                    continue;
                }

                // ---- Forward single new token ----
                $input  = Tensor::fromArray([(float) $next]);
                $logits = $this->model->forwardWithCache($input, $cache);
            }

            // Decode the FULL history (prompt + every generated token)
            return $this->tokenizer->decode($outputIds);
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