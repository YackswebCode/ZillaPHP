<?php

declare(strict_types=1);

namespace ZillaPHP\Transformers;

use ZillaPHP\Attention\MultiHeadAttention;
use ZillaPHP\NN\Layers\FeedForward;
use ZillaPHP\NN\Layers\LayerNorm;
use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

final class TransformerBlock extends Module
{
    private MultiHeadAttention $attn;
    private FeedForward $ffn;
    private LayerNorm $ln1;
    private LayerNorm $ln2;

    public function __construct(int $dim, int $heads, int $ffnHidden)
    {
        $this->attn = new MultiHeadAttention($dim, $heads);
        $this->ffn  = new FeedForward($dim, $ffnHidden);
        $this->ln1  = new LayerNorm($dim);
        $this->ln2  = new LayerNorm($dim);
    }

    public function forward(Tensor $x, ?Tensor $mask = null): Tensor
    {
        $x = $x->add($this->attn->forward($this->ln1->forward($x), $mask));
        $x = $x->add($this->ffn->forward($this->ln2->forward($x)));
        return $x;
    }

    /**
     * Forward pass with KV cache. Used during generation.
     *
     * @param array|null $cache  In/out: ['k' => Tensor, 'v' => Tensor] or null
     */
    public function forwardWithCache(Tensor $x, ?array &$cache): Tensor
    {
        $x = $x->add($this->attn->forwardWithCache($this->ln1->forward($x), $cache));
        $x = $x->add($this->ffn->forward($this->ln2->forward($x)));
        return $x;
    }

    /** @return \ZillaPHP\NN\Parameter[] */
    public function parameters(): array
    {
        return array_merge(
            $this->attn->parameters(),
            $this->ffn->parameters(),
            $this->ln1->parameters(),
            $this->ln2->parameters(),
        );
    }
}