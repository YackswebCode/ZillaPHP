<?php

declare(strict_types=1);

namespace ZillaPHP\NN\Layers;

use ZillaPHP\NN\Module;
use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

final class Linear extends Module
{
    private Parameter $weight;   // [in_features, out_features]
    private ?Parameter $bias;    // [out_features] or null

    public function __construct(
        int $inFeatures,
        int $outFeatures,
        bool $useBias = true,
    ) {
        // Xavier-style scaling:  std = sqrt(1 / in_features)
        $scale = sqrt(1.0 / max($inFeatures, 1));

        $this->weight = new Parameter(
            self::randomNormal([$inFeatures, $outFeatures], $scale),
            new Shape([$inFeatures, $outFeatures])
        );

        $this->bias = $useBias
            ? new Parameter(
                array_fill(0, $outFeatures, 0.0),
                new Shape([$outFeatures])
            )
            : null;
    }

    /**
     * Generate `count` samples from N(0, std) using Box–Muller.
     *
     * @param int[] $shape
     * @return float[]
     */
    private static function randomNormal(array $shape, float $std): array
    {
        $n = array_product($shape);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $u1 = mt_rand() / mt_getrandmax();
            $u2 = mt_rand() / mt_getrandmax();
            $z  = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2);
            $out[] = $z * $std;
        }
        return $out;
    }

    public function forward(Tensor $input): Tensor
    {
        // input: [batch, in_features]
        $out = $input->matmul($this->weight);           // [batch, out_features]
        if ($this->bias !== null) {
            $out = $out->add($this->bias);              // broadcasts [out_features]
        }
        return $out;
    }

    /** @return Parameter[] */
    public function parameters(): array
    {
        return $this->bias === null
            ? [$this->weight]
            : [$this->weight, $this->bias];
    }

    public function weight(): Parameter { return $this->weight; }
    public function bias(): ?Parameter  { return $this->bias; }
}