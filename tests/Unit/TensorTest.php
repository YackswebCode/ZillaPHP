<?php

declare(strict_types=1);

namespace ZillaPHP\Tests\Unit;

use PHPUnit\Framework\TestCase;
use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

final class TensorTest extends TestCase
{
    protected function setUp(): void
    {
        Application::boot();
    }

    public function test_zeros(): void
    {
        $t = Tensor::zeros([2, 3]);
        $this->assertSame([2, 3], $t->shape()->dims());
        $this->assertSame([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], $t->data());
    }

    public function test_add(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $b = Tensor::fromArray([4.0, 5.0, 6.0]);
        $c = $a->add($b);
        $this->assertSame([5.0, 7.0, 9.0], $c->data());
    }

    public function test_matmul(): void
    {
        $a = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $b = Tensor::fromArray([[5.0, 6.0], [7.0, 8.0]]);
        $c = $a->matmul($b);
        $this->assertSame([19.0, 22.0, 43.0, 50.0], $c->data());
    }

    public function test_relu(): void
    {
        $a = Tensor::fromArray([-1.0, 0.0, 2.0, -3.0]);
        $r = $a->relu();
        $this->assertSame([0.0, 0.0, 2.0, 0.0], $r->data());
    }
}