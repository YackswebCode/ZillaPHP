<?php

declare(strict_types=1);

namespace ZillaPHP\Tests\Numerical;

use PHPUnit\Framework\TestCase;
use ZillaPHP\Core\Application;
use ZillaPHP\Tensor\Tensor;

final class GradientCheckTest extends TestCase
{
    protected function setUp(): void
    {
        Application::boot(native: false);
    }

    /**
     * Compare autograd's analytical gradient against central finite differences.
     *
     * Notes on the defaults:
     *   - $eps = 1e-3 is the float32 sweet spot. Larger h reduces float32
     *     cancellation error; smaller h increases it.
     *   - $tol = 1e-2 matches the absolute tolerance PyTorch uses for
     *     float32 gradcheck on values of magnitude ~6.
     *
     * @param callable(Tensor): Tensor $f
     */
    private function checkGradient(
        callable $f,
        float $x0,
        float $eps = 1e-3,
        float $tol = 1e-2,
    ): void {
        // Analytical gradient from autograd
        $x = Tensor::fromArray([$x0], requiresGrad: true);
        $y = $f($x);
        $y->backward();
        $analytic = (float) $x->grad()->item();

        // Numerical gradient via central difference
        $plus  = (float) ($f)(Tensor::fromArray([$x0 + $eps]))->item();
        $minus = (float) ($f)(Tensor::fromArray([$x0 - $eps]))->item();
        $numeric = ($plus - $minus) / (2 * $eps);

        $this->assertEqualsWithDelta(
            $numeric,
            $analytic,
            $tol,
            "Gradient mismatch: analytic={$analytic}, numeric={$numeric}"
        );
    }

    public function test_square(): void
    {
        // f(x) = x²  → f'(x) = 2x
        $this->checkGradient(fn(Tensor $x) => $x->mul($x), 2.0);
        $this->checkGradient(fn(Tensor $x) => $x->mul($x), -3.0);
    }

    public function test_exp(): void
    {
        // f(x) = exp(x) → f'(x) = exp(x)
        $this->checkGradient(fn(Tensor $x) => $x->exp(), 1.5);
    }

    public function test_log(): void
    {
        // f(x) = log(x) → f'(x) = 1/x
        $this->checkGradient(fn(Tensor $x) => $x->log(), 2.0);
    }

    public function test_chain(): void
    {
        // f(x) = log(exp(x) + 1)  (softplus)
        $one = Tensor::fromArray([1.0]);
        $this->checkGradient(
            fn(Tensor $x) => $x->exp()->add($one)->log(),
            1.0
        );
    }

    public function test_relu_positive(): void
    {
        // f(x) = relu(x) at x = 1 → f'(x) = 1
        $this->checkGradient(fn(Tensor $x) => $x->relu(), 1.0);
    }
}