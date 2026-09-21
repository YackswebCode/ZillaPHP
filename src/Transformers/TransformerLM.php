<?php

declare(strict_types=1);

namespace ZillaPHP\Transformers;

use ZillaPHP\NN\Layers\Embedding;
use ZillaPHP\NN\Layers\LayerNorm;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

final class TransformerLM extends Module
{
    private Embedding $tokenEmbed;
    private Embedding $posEmbed;
    /** @var TransformerBlock[] */
    private array $blocks;
    private LayerNorm $lnFinal;
    private Linear $head;

    public function __construct(
        private int $vocabSize,
        private int $dim       = 64,
        int $heads             = 4,
        int $layers            = 2,
        private int $maxSeqLen = 64,
        int $ffnHidden         = 128,
    ) {
        $this->tokenEmbed = new Embedding($vocabSize, $dim);
        $this->posEmbed   = new Embedding($maxSeqLen, $dim);

        $this->blocks = [];
        for ($i = 0; $i < $layers; $i++) {
            $this->blocks[] = new TransformerBlock($dim, $heads, $ffnHidden);
        }

        $this->lnFinal = new LayerNorm($dim);
        $this->head    = new Linear($dim, $vocabSize);
    }

    public function forward(Tensor $tokens): Tensor
    {
        $T = $tokens->shape()->dims()[0];
        if ($T > $this->maxSeqLen) {
            throw new \RuntimeException(
                "Sequence length {$T} exceeds max {$this->maxSeqLen}."
            );
        }

        $x = $this->tokenEmbed->forward($tokens);

        $positions = Tensor::fromArray(range(0, $T - 1));
        $pos = $this->posEmbed->forward($positions);
        $x = $x->add($pos);

        $mask = Tensor::causalMask($T);

        foreach ($this->blocks as $block) {
            $x = $block->forward($x, $mask);
        }

        $x = $this->lnFinal->forward($x);
        return $this->head->forward($x);
    }

    /**
     * Batched forward pass for training.
     *
     * @param Tensor $tokens  Shape [B, T], each element an integer token id.
     * @return Tensor         Shape [B*T, vocab].
     *
     * Layout: batch item b occupies rows b*T .. b*T+T-1.
     * The attention mask is block-diagonal causal, so sequences never
     * attend to each other.
     */
    public function forwardBatch(Tensor $tokens): Tensor
    {
        $dims = $tokens->shape()->dims();
        if (count($dims) !== 2) {
            throw new \RuntimeException(
                "forwardBatch expects [B, T] input; got " . $tokens->shape()
            );
        }
        [$B, $T] = $dims;

        if ($T > $this->maxSeqLen) {
            throw new \RuntimeException(
                "Sequence length {$T} exceeds max {$this->maxSeqLen}."
            );
        }

        // Flatten to [B*T] and embed
        $flat = $tokens->reshape([$B * $T]);
        $x = $this->tokenEmbed->forward($flat);   // [B*T, dim]

        // Positional ids: 0, 1, ..., T-1 repeated B times
        $posIds = [];
        for ($b = 0; $b < $B; $b++) {
            for ($t = 0; $t < $T; $t++) {
                $posIds[] = (float) $t;
            }
        }
        $pos = $this->posEmbed->forward(Tensor::fromArray($posIds));  // [B*T, dim]
        $x = $x->add($pos);

        // Block-diagonal causal mask [B*T, B*T]
        $mask = Tensor::batchCausalMask($B, $T);

        foreach ($this->blocks as $block) {
            $x = $block->forward($x, $mask);
        }

        $x = $this->lnFinal->forward($x);
        return $this->head->forward($x);   // [B*T, vocab]
    }

    /**
     * Forward pass with KV cache for autoregressive generation.
     *
     * @param Tensor  $tokens  Full prompt [T] on first call, or [1] on subsequent calls.
     * @param KVCache $cache   In/out cache object.
     */
    public function forwardWithCache(Tensor $tokens, KVCache $cache): Tensor
    {
        $T      = $tokens->shape()->dims()[0];
        $offset = $cache->length;

        if ($offset + $T > $this->maxSeqLen) {
            throw new \RuntimeException(
                "Sequence position " . ($offset + $T) . " exceeds max {$this->maxSeqLen}."
            );
        }

        $x = $this->tokenEmbed->forward($tokens);

        $positions = Tensor::fromArray(array_map('floatval', range($offset, $offset + $T - 1)));
        $pos = $this->posEmbed->forward($positions);
        $x = $x->add($pos);

        foreach ($this->blocks as $i => $block) {
            $layerCache = $cache->layers[$i] ?? null;
            $x = $block->forwardWithCache($x, $layerCache);
            $cache->layers[$i] = $layerCache;
        }

        $cache->length += $T;

        $x = $this->lnFinal->forward($x);
        return $this->head->forward($x);
    }

    public function newCache(): KVCache
    {
        return new KVCache(count($this->blocks));
    }

    /** @return \ZillaPHP\NN\Parameter[] */
    public function parameters(): array
    {
        $params = array_merge(
            $this->tokenEmbed->parameters(),
            $this->posEmbed->parameters(),
            $this->lnFinal->parameters(),
            $this->head->parameters(),
        );
        foreach ($this->blocks as $b) {
            $params = array_merge($params, $b->parameters());
        }
        return $params;
    }

    public function maxSeqLen(): int { return $this->maxSeqLen; }
    public function vocabSize(): int { return $this->vocabSize; }
    public function dim(): int       { return $this->dim; }
}