<?php

declare(strict_types=1);

namespace ZillaPHP\Core\Contracts;

use ZillaPHP\Tensor\Tensor;

interface Backend
{
    public function name(): string;

    public function add(Tensor $a, Tensor $b): Tensor;
    public function subtract(Tensor $a, Tensor $b): Tensor;
    public function multiply(Tensor $a, Tensor $b): Tensor;
    public function divide(Tensor $a, Tensor $b): Tensor;
    public function matmul(Tensor $a, Tensor $b): Tensor;

    public function sum(Tensor $tensor): Tensor;
    public function mean(Tensor $tensor): Tensor;

    public function relu(Tensor $tensor): Tensor;
    public function exp(Tensor $tensor): Tensor;
    public function log(Tensor $tensor): Tensor;
}