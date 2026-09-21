<?php

declare(strict_types=1);

namespace ZillaPHP\Attention;

use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

/**
 * Multi-head self-attention.
 *
 *   input : Tensor [T, dim]
 *   output: Tensor [T, dim]
 *
 * Each of the $numHeads heads operates on dim/numHeads columns,
 * runs scaled dot-product attention independently, and the results
 * are concatenated and passed through an output projection.
 */
final class MultiHeadAttention extends Module
{
    private Linear $wq;
    private Linear $wk;
    private Linear $wv;
    private Linear $wo;

    public function __construct(private int $dim, private int $numHeads)
    {
        if ($dim % $numHeads !== 0) {
            throw new \InvalidArgumentException(
                "dim ({$dim}) must be divisible by numHeads ({$numHeads})."
            );
        }
        $this->wq = new Linear($dim, $dim);
        $this->wk = new Linear($dim, $dim);
        $this->wv = new Linear($dim, $dim);
        $this->wo = new Linear($dim, $dim);
    }

    public function forward(Tensor $x, ?Tensor $mask = null): Tensor
    {
        // Project to Q, K, V — each [T, dim]
        $q = $this->wq->forward($x);
        $k = $this->wk->forward($x);
        $v = $this->wv->forward($x);

        // Split into heads — each [T, headDim]
        $qs = $q->chunkColumns($this->numHeads);
        $ks = $k->chunkColumns($this->numHeads);
        $vs = $v->chunkColumns($this->numHeads);

        $outs = [];
        for ($h = 0; $h < $this->numHeads; $h++) {
            $outs[] = $this->attendHead($qs[$h], $ks[$h], $vs[$h], $mask);
        }

        // Concatenate back and project — [T, dim]
        $concat = Tensor::concatColumns($outs);
        return $this->wo->forward($concat);
    }

    private function attendHead(Tensor $q, Tensor $k, Tensor $v, ?Tensor $mask): Tensor
    {
        $headDim = $q->shape()->dims()[1];
        $scale   = 1.0 / sqrt($headDim);

        // scores = Q @ K^T / sqrt(d)   → [T, T]
        $scores = $q->matmul($k->transpose());
        $scores = $scores->mul(Tensor::full([1], $scale));

        // Apply mask (add -1e9 to future positions) — softmax zeroes them out
        if ($mask !== null) {
            $scores = $scores->maskedFill($mask, -1e9);
        }

        // Softmax over keys, then weight values
        $attn = $scores->softmax();
        return $attn->matmul($v);
    }

    /** @return \ZillaPHP\NN\Parameter[] */
    public function parameters(): array
    {
        return array_merge(
            $this->wq->parameters(),
            $this->wk->parameters(),
            $this->wv->parameters(),
            $this->wo->parameters(),
        );
    }
}