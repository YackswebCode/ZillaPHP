<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Module;
use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Shape\Shape;   // ✓ correct
use ZillaPHP\Tensor\Tensor;

/**
 * Token embedding.
 *
 *   input : Tensor [seq_len]   (each element is a token id in [0, vocab))
 *   output: Tensor [seq_len, embed_dim]
 */
final class Embedding extends Module
{
    private Parameter $table;   // [vocab_size, embed_dim]

    public function __construct(private int $vocabSize, private int $embedDim)
    {
        $scale = 0.01;
        $n = $vocabSize * $embedDim;
        $data = [];
        for ($i = 0; $i < $n; $i++) {
            $u1 = mt_rand() / mt_getrandmax();
            $u2 = mt_rand() / mt_getrandmax();
            $data[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2) * $scale;
        }
        $this->table = new Parameter($data, new Shape([$vocabSize, $embedDim]));
    }

    public function forward(Tensor $input): Tensor
    {
        $ids    = $input->data();
        $seqLen = count($ids);
        $dims   = $this->table->shape()->dims();
        [$vocab, $dim] = $dims;

        // One-hot matmul so autograd flows through the table automatically.
        $oneHot = [];
        for ($i = 0; $i < $seqLen; $i++) {
            $id = (int) $ids[$i];
            if ($id < 0 || $id >= $vocab) {
                throw new \OutOfRangeException("Token id {$id} out of range [0, {$vocab}).");
            }
            for ($j = 0; $j < $vocab; $j++) {
                $oneHot[] = ($j === $id) ? 1.0 : 0.0;
            }
        }

        $oneHotT = new Tensor($oneHot, new Shape([$seqLen, $vocab]));
        return $oneHotT->matmul($this->table);
    }

    /** @return Parameter[] */
    public function parameters(): array
    {
        return [$this->table];
    }

    public function table(): Parameter { return $this->table; }
    public function embedDim(): int    { return $this->embedDim; }
    public function vocabSize(): int   { return $this->vocabSize; }
}