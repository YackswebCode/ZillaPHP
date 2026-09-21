<?php

declare(strict_types=1);

namespace ZillaPHP\Transformers;

use ZillaPHP\NN\Layers\Embedding;
use ZillaPHP\NN\Layers\LayerNorm;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

/**
 * A small autoregressive language model.
 *
 *   input : Tensor [T]        integer token ids
 *   output: Tensor [T, vocab] logits for next token at each position
 */
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

        // Token + positional embeddings — [T, dim]
        $x = $this->tokenEmbed->forward($tokens);

        $positions = Tensor::fromArray(range(0, $T - 1));
        $pos = $this->posEmbed->forward($positions);
        $x = $x->add($pos);

        // Causal mask — [T, T]
        $mask = Tensor::causalMask($T);

        foreach ($this->blocks as $block) {
            $x = $block->forward($x, $mask);
        }

        $x = $this->lnFinal->forward($x);
        return $this->head->forward($x);   // [T, vocab]
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