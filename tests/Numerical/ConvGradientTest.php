<?php

declare(strict_types=1);

namespace ZillaPHP\Tests\Numerical;

use PHPUnit\Framework\TestCase;
use ZillaPHP\Core\Application;
use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

final class ConvGradientTest extends TestCase
{
    protected function setUp(): void
    {
        Application::boot(native: false);
    }

    public function test_conv2d_input_gradient(): void
    {
        $inC = 1; $H = 4; $W = 4;
        $outC = 2; $kH = 3; $kW = 3;

        $inputData = [];
        for ($i = 0; $i < $inC * $H * $W; $i++) {
            $inputData[] = (mt_rand() / mt_getrandmax()) * 2 - 1;
        }
        $wData = [];
        for ($i = 0; $i < $outC * $inC * $kH * $kW; $i++) {
            $wData[] = (mt_rand() / mt_getrandmax()) * 2 - 1;
        }
        $bData = [0.1, -0.2];

        // Analytical gradient
        $input  = new Tensor($inputData, new Shape([$inC, $H, $W]), requiresGrad: true);
        $weight = new Tensor($wData, new Shape([$outC, $inC, $kH, $kW]));
        $bias   = new Tensor($bData, new Shape([$outC]));

        $out = $input->conv2d($weight, $bias, stride: 1, padding: 1);
        $loss = $out->mul($out)->sum();
        $loss->backward();
        $analytic = $input->grad()->data();

        // Numerical
        $eps = 1e-3;
        $numeric = [];
        for ($i = 0; $i < count($inputData); $i++) {
            $plus  = $inputData; $plus[$i]  += $eps;
            $minus = $inputData; $minus[$i] -= $eps;

            $lp = (new Tensor($plus,  new Shape([$inC, $H, $W])))
                ->conv2d($weight, $bias, 1, 1)
                ->mul((new Tensor($plus,  new Shape([$inC, $H, $W])))
                    ->conv2d($weight, $bias, 1, 1))
                ->sum()->item();

            $lm = (new Tensor($minus, new Shape([$inC, $H, $W])))
                ->conv2d($weight, $bias, 1, 1)
                ->mul((new Tensor($minus, new Shape([$inC, $H, $W])))
                    ->conv2d($weight, $bias, 1, 1))
                ->sum()->item();

            $numeric[] = ($lp - $lm) / (2 * $eps);
        }

        $maxErr = 0.0;
        foreach ($analytic as $i => $a) {
            $err = abs($a - $numeric[$i]);
            if ($err > $maxErr) $maxErr = $err;
        }

        $this->assertLessThan(1e-2, $maxErr,
            "conv2d input gradient max error: {$maxErr}");
    }

    public function test_conv2d_weight_gradient(): void
    {
        $inC = 1; $H = 4; $W = 4; $outC = 2; $kH = 3; $kW = 3;

        $inputData = [];
        for ($i = 0; $i < $inC * $H * $W; $i++) $inputData[] = (mt_rand() / mt_getrandmax()) * 2 - 1;
        $wData = [];
        for ($i = 0; $i < $outC * $inC * $kH * $kW; $i++) $wData[] = (mt_rand() / mt_getrandmax()) * 2 - 1;
        $bData = [0.0, 0.0];

        $input  = new Tensor($inputData, new Shape([$inC, $H, $W]));
        $weight = new Tensor($wData, new Shape([$outC, $inC, $kH, $kW]), requiresGrad: true);
        $bias   = new Tensor($bData, new Shape([$outC]));

        $out  = $input->conv2d($weight, $bias, 1, 1);
        $loss = $out->mul($out)->sum();
        $loss->backward();
        $analytic = $weight->grad()->data();

        $eps = 1e-3;
        $numeric = [];
        foreach ($wData as $i => $_) {
            $plus  = $wData; $plus[$i]  += $eps;
            $minus = $wData; $minus[$i] -= $eps;

            $lp = ($input->conv2d(new Tensor($plus, new Shape([$outC, $inC, $kH, $kW])), $bias, 1, 1))
                ->mul($input->conv2d(new Tensor($plus, new Shape([$outC, $inC, $kH, $kW])), $bias, 1, 1))
                ->sum()->item();

            $lm = ($input->conv2d(new Tensor($minus, new Shape([$outC, $inC, $kH, $kW])), $bias, 1, 1))
                ->mul($input->conv2d(new Tensor($minus, new Shape([$outC, $inC, $kH, $kW])), $bias, 1, 1))
                ->sum()->item();

            $numeric[] = ($lp - $lm) / (2 * $eps);
        }

        $maxErr = 0.0;
        foreach ($analytic as $i => $a) {
            $err = abs($a - $numeric[$i]);
            if ($err > $maxErr) $maxErr = $err;
        }

        $this->assertLessThan(1e-2, $maxErr,
            "conv2d weight gradient max error: {$maxErr}");
    }
}