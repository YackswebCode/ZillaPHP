<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Module;
use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Shape\Shape;   // ✓ correct
use ZillaPHP\Tensor\Tensor;

/**
 * Layer normalization over the last dimension.
 *
 *   input : Tensor [seq_len, dim]
 *   output: Tensor [seq_len, dim]
 */
final class LayerNorm extends Module
{
    private Parameter $gamma;
    private Parameter $beta;

    public function __construct(private int $dim, private float $eps = 1e-5)
    {
        $this->gamma = new Parameter(array_fill(0, $dim, 1.0), new Shape([$dim]));
        $this->beta  = new Parameter(array_fill(0, $dim, 0.0), new Shape([$dim]));
    }

        public function forward(Tensor $input): Tensor
    {
        return $input->layerNorm($this->gamma, $this->beta, $this->eps);
    }

    /** @return Parameter[] */
    public function parameters(): array
    {
        return [$this->gamma, $this->beta];
    }
}