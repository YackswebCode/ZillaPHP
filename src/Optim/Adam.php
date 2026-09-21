<?php

declare(strict_types=1);

namespace ZillaPHP\Optim;

use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Tensor;

final class Adam extends Optimizer
{
    private float $beta1;
    private float $beta2;
    private float $eps;
    private int $t = 0;

    /** @var array<int, array<int,float>> */
    private array $m = [];
    /** @var array<int, array<int,float>> */
    private array $v = [];

    public function __construct(array $params, float $lr = 0.001, float $beta1 = 0.9, float $beta2 = 0.999, float $eps = 1e-8)
    {
        parent::__construct($params, $lr);
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->eps   = $eps;

        foreach ($params as $p) {
            $id = spl_object_id($p);
            $this->m[$id] = array_fill(0, $p->shape()->size(), 0.0);
            $this->v[$id] = array_fill(0, $p->shape()->size(), 0.0);
        }
    }

    public function step(): void
    {
        $this->t++;
        $bc1 = 1.0 - ($this->beta1 ** $this->t);
        $bc2 = 1.0 - ($this->beta2 ** $this->t);

        foreach ($this->params as $p) {
            $g = $p->grad();
            if ($g === null) continue;

            $id = spl_object_id($p);
            $gd = $g->data();
            $pd = $p->data();
            $new = [];

            foreach ($pd as $i => $theta) {
                $this->m[$id][$i] = $this->beta1 * $this->m[$id][$i] + (1.0 - $this->beta1) * $gd[$i];
                $this->v[$id][$i] = $this->beta2 * $this->v[$id][$i] + (1.0 - $this->beta2) * ($gd[$i] ** 2);

                $mHat = $this->m[$id][$i] / $bc1;
                $vHat = $this->v[$id][$i] / $bc2;

                $new[] = $theta - $this->lr * $mHat / (sqrt($vHat) + $this->eps);
            }
            $this->setData($p, $new);
        }
    }

    private function setData(Parameter $p, array $data): void
    {
        $ref = new \ReflectionClass(Tensor::class);
        $prop = $ref->getProperty('data');
        $prop->setAccessible(true);
        $prop->setValue($p, $data);
    }
}