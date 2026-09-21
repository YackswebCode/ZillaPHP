<?php

declare(strict_types=1);

namespace ZillaPHP\Tensor;

use ZillaPHP\Core\Contracts\Backend;
use ZillaPHP\Tensor\Device\Device;
use ZillaPHP\Tensor\DType\DType;
use ZillaPHP\Tensor\Shape\Shape;

class Tensor
{
    /** @var array<int|float|bool> */
    protected array $data;
    protected Shape $shape;
    protected DType $dtype;
    protected Device $device;

    protected bool $requiresGrad = false;
    protected ?Tensor $grad = null;
    /** @var Tensor[] */
    protected array $inputs = [];
    /** @var callable|null */
    protected $backwardFn = null;
    protected string $op = '';

    private static int $noGradDepth = 0;

    /** @param array<int|float|bool> $data */
    public function __construct(
        array $data,
        Shape $shape,
        DType $dtype = DType::FLOAT32,
        ?Device $device = null,
        bool $requiresGrad = false,
    ) {
        if (count($data) !== $shape->size()) {
            throw new \InvalidArgumentException(
                "Data size (" . count($data) . ") does not match shape size (" . $shape->size() . ")."
            );
        }
        $this->data         = $data;
        $this->shape        = $shape;
        $this->dtype        = $dtype;
        $this->device       = $device ?? Device::cpu();
        $this->requiresGrad = $requiresGrad;
    }

    // ==================================================================
    // No-grad mode
    // ==================================================================

    public static function noGrad(callable $fn): mixed
    {
        self::$noGradDepth++;
        try {
            return $fn();
        } finally {
            self::$noGradDepth--;
        }
    }

    public static function gradEnabled(): bool
    {
        return self::$noGradDepth === 0;
    }

    // ==================================================================
    // Factories
    // ==================================================================

    public static function zeros(array $shape, bool $requiresGrad = false): self
    {
        $s = new Shape($shape);
        return new self(array_fill(0, $s->size(), 0.0), $s, DType::FLOAT32, null, $requiresGrad);
    }

    public static function ones(array $shape, bool $requiresGrad = false): self
    {
        $s = new Shape($shape);
        return new self(array_fill(0, $s->size(), 1.0), $s, DType::FLOAT32, null, $requiresGrad);
    }

    public static function full(array $shape, int|float $value, bool $requiresGrad = false): self
    {
        $s = new Shape($shape);
        return new self(array_fill(0, $s->size(), $value), $s, DType::FLOAT32, null, $requiresGrad);
    }

    public static function randn(array $shape, bool $requiresGrad = false): self
    {
        $s = new Shape($shape);
        $data = [];
        for ($i = 0; $i < $s->size(); $i++) {
            $u1 = mt_rand() / mt_getrandmax();
            $u2 = mt_rand() / mt_getrandmax();
            $data[] = sqrt(-2.0 * log($u1 ?: 1e-12)) * cos(2.0 * M_PI * $u2);
        }
        return new self($data, $s, DType::FLOAT32, null, $requiresGrad);
    }

    public static function fromArray(array $data, bool $requiresGrad = false): self
    {
        $flat = [];
        $shape = self::inferShapeAndFlatten($data, $flat);
        return new self($flat, new Shape($shape), DType::FLOAT32, null, $requiresGrad);
    }

    /**
     * @param Tensor[] $tensors
     */
    public static function concatRows(array $tensors): self
    {
        if (empty($tensors)) {
            throw new \InvalidArgumentException("concatRows requires at least one tensor.");
        }

        $first = $tensors[0]->shape()->dims();
        if (count($first) !== 2 || $first[0] !== 1) {
            throw new \InvalidArgumentException(
                "concatRows expects each tensor to have shape [1, D]; got [" .
                implode(',', $first) . "]"
            );
        }
        $d = $first[1];

        $out = [];
        foreach ($tensors as $t) {
            if ($t->shape()->dims() !== [1, $d]) {
                throw new \InvalidArgumentException("concatRows: inconsistent row size.");
            }
            foreach ($t->data() as $v) {
                $out[] = $v;
            }
        }

        return new self($out, new Shape([count($tensors), $d]));
    }

    /**
     * @param Tensor[] $tensors
     */
    public static function concatAlongRows(array $tensors): self
    {
        if (empty($tensors)) {
            throw new \InvalidArgumentException("concatAlongRows requires tensors.");
        }

        $first = $tensors[0]->shape()->dims();
        if (count($first) !== 2) {
            throw new \RuntimeException("concatAlongRows requires 2D tensors.");
        }
        $cols = $first[1];

        $rowsTotal = 0;
        foreach ($tensors as $t) {
            $td = $t->shape()->dims();
            if (count($td) !== 2 || $td[1] !== $cols) {
                throw new \InvalidArgumentException("concatAlongRows: column mismatch.");
            }
            $rowsTotal += $td[0];
        }

        $out = [];
        foreach ($tensors as $t) {
            foreach ($t->data() as $v) {
                $out[] = $v;
            }
        }

        return new self($out, new Shape([$rowsTotal, $cols]));
    }

    /** @param array<mixed> $data @param array<int|float|bool> $flat @return int[] */
    private static function inferShapeAndFlatten(array $data, array &$flat): array
    {
        $count = count($data);
        if ($count === 0) return [0];

        if (!is_array($data[0])) {
            foreach ($data as $v) {
                if (is_array($v)) throw new \InvalidArgumentException('Ragged arrays are not supported.');
                $flat[] = $v;
            }
            return [$count];
        }
        $sub = null;
        foreach ($data as $child) {
            if (!is_array($child)) throw new \InvalidArgumentException('Ragged arrays are not supported.');
            $cs = self::inferShapeAndFlatten($child, $flat);
            if ($sub === null) $sub = $cs;
            elseif ($sub !== $cs) throw new \InvalidArgumentException('Ragged arrays are not supported.');
        }
        return array_merge([$count], $sub ?? []);
    }

    // ==================================================================
    // Accessors
    // ==================================================================

    public function shape(): Shape       { return $this->shape; }
    public function dtype(): DType       { return $this->dtype; }
    public function device(): Device     { return $this->device; }
    public function data(): array        { return $this->data; }
    public function requiresGrad(): bool { return $this->requiresGrad; }
    public function grad(): ?Tensor      { return $this->grad; }
    public function op(): string         { return $this->op; }

    public function item(): int|float|bool
    {
        if ($this->shape->size() !== 1) throw new \RuntimeException("item() requires a scalar.");
        return $this->data[0];
    }

    public function zeroGrad(): void { $this->grad = null; }

    // ==================================================================
    // Backend
    // ==================================================================

    private static ?Backend $backend = null;

    public static function setBackend(Backend $b): void { self::$backend = $b; }

    public static function backend(): Backend
    {
        if (self::$backend === null) throw new \RuntimeException("No backend registered.");
        return self::$backend;
    }

    // ==================================================================
    // Autograd registration
    // ==================================================================

    /**
     * @param Tensor[] $inputs
     */
    public static function attachAutograd(
        Tensor $result,
        array $inputs,
        string $op,
        callable $backwardFn,
    ): Tensor {
        if (!self::gradEnabled()) {
            return $result;
        }
        $result->requiresGrad = true;
        $result->op           = $op;
        $result->inputs       = $inputs;
        $result->backwardFn   = $backwardFn;
        return $result;
    }

    public function accumulateGradPublic(Tensor $g): void
    {
        $this->accumulateGrad($g);
    }

    // ==================================================================
    // Broadcasting helpers
    // ==================================================================

    /** @return int[] */
    private static function broadcastShapes(array $a, array $b): array
    {
        $n  = max(count($a), count($b));
        $pa = array_merge(array_fill(0, $n - count($a), 1), $a);
        $pb = array_merge(array_fill(0, $n - count($b), 1), $b);
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            if ($pa[$i] === $pb[$i])    $out[] = $pa[$i];
            elseif ($pa[$i] === 1)      $out[] = $pb[$i];
            elseif ($pb[$i] === 1)      $out[] = $pa[$i];
            else throw new \InvalidArgumentException(
                "Cannot broadcast [" . implode(',', $a) . "] with [" . implode(',', $b) . "]"
            );
        }
        return $out;
    }

    /** @return int[] */
    private static function unravelIndex(int $flat, array $shape): array
    {
        $n = count($shape);
        $idx = array_fill(0, $n, 0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $idx[$i] = $flat % $shape[$i];
            $flat = intdiv($flat, $shape[$i]);
        }
        return $idx;
    }

    /** @param int[] $idx @param int[] $shape */
    private static function ravelIndex(array $idx, array $shape): int
    {
        $flat = 0;
        foreach ($shape as $i => $dim) $flat = $flat * $dim + $idx[$i];
        return $flat;
    }

    /** @param int[] $target */
    private static function broadcastTo(Tensor $t, array $target): Tensor
    {
        $from = $t->shape()->dims();
        if ($from === $target) return $t;

        $n = count($target);
        $fromPad = array_merge(array_fill(0, $n - count($from), 1), $from);
        $data = $t->data();
        $total = array_product($target);
        $out = [];

        for ($flat = 0; $flat < $total; $flat++) {
            $idx = self::unravelIndex($flat, $target);
            $srcIdx = [];
            for ($i = 0; $i < $n; $i++) $srcIdx[] = ($fromPad[$i] === 1) ? 0 : $idx[$i];
            $out[] = $data[self::ravelIndex($srcIdx, $fromPad)];
        }
        return new Tensor($out, new Shape($target));
    }

    /** @param int[] $targetShape */
    private static function reduceGrad(Tensor $grad, array $targetShape): Tensor
    {
        $srcShape = $grad->shape()->dims();
        if ($srcShape === $targetShape) return $grad;

        $n = count($srcShape);
        $targetPad = array_merge(array_fill(0, $n - count($targetShape), 1), $targetShape);
        $targetSize = array_product($targetShape);
        $out = array_fill(0, $targetSize, 0.0);

        $total = array_product($srcShape);
        $gd = $grad->data();

        for ($flat = 0; $flat < $total; $flat++) {
            $idx = self::unravelIndex($flat, $srcShape);
            $tIdx = [];
            for ($i = 0; $i < $n; $i++) $tIdx[] = ($targetPad[$i] === 1) ? 0 : $idx[$i];
            $tFlat = self::ravelIndex($tIdx, $targetPad);
            $out[$tFlat] += $gd[$flat];
        }
        return new Tensor($out, new Shape($targetShape));
    }

    // ==================================================================
    // Autograd core
    // ==================================================================

    public function backward(?Tensor $gradient = null): void
    {
        if (!$this->requiresGrad) throw new \RuntimeException("backward() called on non-grad tensor.");
        if ($gradient === null) {
            if ($this->shape->size() !== 1) throw new \RuntimeException(
                "backward() without gradient requires a scalar tensor."
            );
            $gradient = self::ones($this->shape->dims());
        }

        $topo = []; $visited = [];
        $this->buildTopo($topo, $visited);
        $this->grad = $gradient;

        foreach (array_reverse($topo) as $node) {
            if ($node->backwardFn !== null && $node->grad !== null) {
                ($node->backwardFn)($node->grad);
            }
        }
    }

    /** @param Tensor[] $topo @param array<int,bool> $visited */
    private function buildTopo(array &$topo, array &$visited): void
    {
        $id = spl_object_id($this);
        if (isset($visited[$id])) return;
        $visited[$id] = true;
        foreach ($this->inputs as $i) $i->buildTopo($topo, $visited);
        $topo[] = $this;
    }

    protected function accumulateGrad(Tensor $g): void
    {
        $this->grad = $this->grad === null
            ? $g
            : self::backend()->add($this->grad, $g);
    }

    // ==================================================================
    // Shape manipulation with autograd
    // ==================================================================

    public function reshape(array $newShape): self
    {
        $target = 1;
        foreach ($newShape as $d) $target *= $d;
        if ($target !== $this->shape->size()) {
            throw new \InvalidArgumentException(
                "Cannot reshape from {$this->shape} to [" . implode(',', $newShape) .
                "] — element count {$target} vs {$this->shape->size()}"
            );
        }

        $result = new self($this->data, new Shape($newShape), $this->dtype, $this->device);

        if (self::gradEnabled() && $this->requiresGrad) {
            $origShape = $this->shape->dims();
            self::attachAutograd(
                $result,
                [$this],
                'reshape',
                function (Tensor $g) use ($result, $origShape): void {
                    $input = $result->inputs[0];
                    $input->accumulateGradPublic(
                        new self($g->data(), new Shape($origShape))
                    );
                },
            );
        }

        return $result;
    }

    public function sliceRows(int $start, int $end): self
    {
        $dims = $this->shape->dims();
        if (count($dims) !== 2) {
            throw new \RuntimeException("sliceRows requires 2D tensor.");
        }
        [$rows, $cols] = $dims;
        if ($start < 0 || $end > $rows || $start >= $end) {
            throw new \OutOfRangeException(
                "Invalid slice [{$start}, {$end}) for tensor with {$rows} rows."
            );
        }

        $out = [];
        for ($i = $start; $i < $end; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $out[] = $this->data[$i * $cols + $j];
            }
        }

        return new self($out, new Shape([$end - $start, $cols]), $this->dtype, $this->device);
    }

    /**
     * @return Tensor[]
     */
    public function chunkColumns(int $n): array
    {
        $dims = $this->shape->dims();
        if (count($dims) !== 2) throw new \RuntimeException("chunkColumns requires 2D tensor.");
        [$rows, $cols] = $dims;
        if ($cols % $n !== 0) {
            throw new \InvalidArgumentException(
                "Cannot split {$cols} columns into {$n} equal chunks."
            );
        }
        $chunkSize = intdiv($cols, $n);
        $out = [];

        for ($c = 0; $c < $n; $c++) {
            $piece = [];
            for ($i = 0; $i < $rows; $i++) {
                for ($j = 0; $j < $chunkSize; $j++) {
                    $piece[] = $this->data[$i * $cols + $c * $chunkSize + $j];
                }
            }
            $chunk = new self($piece, new Shape([$rows, $chunkSize]), $this->dtype, $this->device);

            if (self::gradEnabled() && $this->requiresGrad) {
                $source = $this;
                $chunkC = $c;
                Tensor::attachAutograd(
                    $chunk,
                    [$this],
                    "chunk_columns_{$c}",
                    function (Tensor $g) use ($source, $chunkC, $chunkSize, $cols, $rows): void {
                        $full = array_fill(0, $rows * $cols, 0.0);
                        $gd = $g->data();
                        for ($i = 0; $i < $rows; $i++) {
                            for ($j = 0; $j < $chunkSize; $j++) {
                                $full[$i * $cols + $chunkC * $chunkSize + $j] = $gd[$i * $chunkSize + $j];
                            }
                        }
                        $source->accumulateGradPublic(new self($full, $source->shape()));
                    },
                );
            }
            $out[] = $chunk;
        }
        return $out;
    }

    /**
     * @param Tensor[] $tensors
     */
    public static function concatColumns(array $tensors): self
    {
        if (empty($tensors)) throw new \InvalidArgumentException("concatColumns requires tensors.");
        $rows = $tensors[0]->shape()->dims()[0] ?? 0;
        $colsTotal = 0;
        $colWidths = [];
        foreach ($tensors as $t) {
            if ($t->shape()->dims()[0] !== $rows) {
                throw new \InvalidArgumentException("concatColumns: row count mismatch.");
            }
            $cw = $t->shape()->dims()[1];
            $colWidths[] = $cw;
            $colsTotal += $cw;
        }

        $out = [];
        for ($i = 0; $i < $rows; $i++) {
            foreach ($tensors as $t) {
                $td = $t->data();
                $cw = $t->shape()->dims()[1];
                for ($j = 0; $j < $cw; $j++) {
                    $out[] = $td[$i * $cw + $j];
                }
            }
        }

        $result = new self($out, new Shape([$rows, $colsTotal]));

        if (self::gradEnabled()) {
            $anyGrad = false;
            foreach ($tensors as $t) {
                if ($t->requiresGrad) { $anyGrad = true; break; }
            }

            if ($anyGrad) {
                Tensor::attachAutograd(
                    $result,
                    $tensors,
                    'concat_columns',
                    function (Tensor $g) use ($tensors, $rows, $colsTotal, $colWidths): void {
                        $gd = $g->data();
                        $colOffset = 0;
                        foreach ($tensors as $idx => $t) {
                            $cw = $colWidths[$idx];
                            if ($t->requiresGrad) {
                                $part = [];
                                for ($i = 0; $i < $rows; $i++) {
                                    for ($j = 0; $j < $cw; $j++) {
                                        $part[] = $gd[$i * $colsTotal + $colOffset + $j];
                                    }
                                }
                                $t->accumulateGradPublic(new self($part, $t->shape()));
                            }
                            $colOffset += $cw;
                        }
                    },
                );
            }
        }
        return $result;
    }

    // ==================================================================
    // Masking / LayerNorm
    // ==================================================================

    public function maskedFill(Tensor $mask, float $value = -1e9): self
    {
        if ($mask->shape()->dims() !== $this->shape->dims()) {
            throw new \InvalidArgumentException(
                "maskedFill: mask shape {$mask->shape()} must match tensor shape {$this->shape}"
            );
        }
        $n   = $this->shape->size();
        $m   = $mask->data();
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = $m[$i] > 0.5 ? $value : $this->data[$i];
        }

        $result = new self($out, $this->shape, $this->dtype, $this->device);
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true;
            $result->op = 'masked_fill';
            $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result, $mask): void {
                $a  = $result->inputs[0];
                $gd = $g->data();
                $md = $mask->data();
                $out = [];
                foreach ($gd as $i => $v) $out[] = $md[$i] > 0.5 ? 0.0 : $v;
                $a->accumulateGrad(new self($out, $g->shape()));
            };
        }
        return $result;
    }

    public static function causalMask(int $size, float $above = 1.0): self
    {
        $data = [];
        for ($i = 0; $i < $size; $i++) {
            for ($j = 0; $j < $size; $j++) {
                $data[] = ($j > $i) ? $above : 0.0;
            }
        }
        return new self($data, new Shape([$size, $size]));
    }

    public function layerNorm(Tensor $gamma, Tensor $beta, float $eps = 1e-5): self
    {
        $dims = $this->shape->dims();
        if (count($dims) !== 2) {
            throw new \RuntimeException("layerNorm requires 2D input.");
        }
        [$rows, $dim] = $dims;

        if ($gamma->shape()->dims() !== [$dim] || $beta->shape()->dims() !== [$dim]) {
            throw new \InvalidArgumentException("gamma and beta must have shape [{$dim}].");
        }

        $d  = $this->data();
        $gd = $gamma->data();
        $bd = $beta->data();

        $invStds = [];
        $xhats   = [];
        $out     = [];

        for ($i = 0; $i < $rows; $i++) {
            $offset = $i * $dim;

            $mean = 0.0;
            for ($j = 0; $j < $dim; $j++) $mean += $d[$offset + $j];
            $mean /= $dim;

            $var = 0.0;
            for ($j = 0; $j < $dim; $j++) {
                $diff = $d[$offset + $j] - $mean;
                $var += $diff * $diff;
            }
            $var /= $dim;
            $invStd = 1.0 / sqrt($var + $eps);
            $invStds[$i] = $invStd;

            for ($j = 0; $j < $dim; $j++) {
                $xhat = ($d[$offset + $j] - $mean) * $invStd;
                $xhats[$offset + $j] = $xhat;
                $out[] = $xhat * $gd[$j] + $bd[$j];
            }
        }

        $result = new self($out, $this->shape, $this->dtype, $this->device);

        if (self::gradEnabled()
            && ($this->requiresGrad || $gamma->requiresGrad || $beta->requiresGrad)
        ) {
            self::attachAutograd(
                $result,
                [$this, $gamma, $beta],
                'layer_norm',
                function (Tensor $g) use ($result, $gd, $xhats, $invStds, $rows, $dim): void {
                    [$input, $gamma, $beta] = $result->inputs;
                    $gData = $g->data();

                    $gradGamma = array_fill(0, $dim, 0.0);
                    $gradBeta  = array_fill(0, $dim, 0.0);
                    $gradInput = array_fill(0, $rows * $dim, 0.0);

                    for ($i = 0; $i < $rows; $i++) {
                        $offset = $i * $dim;
                        $invStd = $invStds[$i];

                        $sumGxhat     = 0.0;
                        $sumGxhatXhat = 0.0;
                        $gxhat        = [];

                        for ($j = 0; $j < $dim; $j++) {
                            $gxhat[$j] = $gData[$offset + $j] * $gd[$j];
                            $xhat      = $xhats[$offset + $j];
                            $gradGamma[$j] += $gData[$offset + $j] * $xhat;
                            $gradBeta[$j]  += $gData[$offset + $j];
                            $sumGxhat     += $gxhat[$j];
                            $sumGxhatXhat += $gxhat[$j] * $xhat;
                        }

                        for ($j = 0; $j < $dim; $j++) {
                            $xhat = $xhats[$offset + $j];
                            $gradInput[$offset + $j] = $invStd / $dim * (
                                $dim * $gxhat[$j] - $sumGxhat - $xhat * $sumGxhatXhat
                            );
                        }
                    }

                    if ($input->requiresGrad) {
                        $input->accumulateGradPublic(new self($gradInput, $input->shape()));
                    }
                    if ($gamma->requiresGrad) {
                        $gamma->accumulateGradPublic(new self($gradGamma, $gamma->shape()));
                    }
                    if ($beta->requiresGrad) {
                        $beta->accumulateGradPublic(new self($gradBeta, $beta->shape()));
                    }
                },
            );
        }

        return $result;
    }

    // ==================================================================
    // Vision ops
    // ==================================================================

    /**
     * 2D convolution. Uses the native kernel when available.
     */
    public function conv2d(
        Tensor $weight,
        Tensor $bias,
        int $stride = 1,
        int $padding = 0,
    ): self {
        $inShape  = $this->shape->dims();
        $wShape   = $weight->shape()->dims();

        if (count($inShape) !== 3) {
            throw new \RuntimeException("conv2d input must be [C, H, W].");
        }
        if (count($wShape) !== 4) {
            throw new \RuntimeException("conv2d weight must be [outC, C, kH, kW].");
        }

        [$C, $H, $W]           = $inShape;
        [$outC, $Cw, $kH, $kW] = $wShape;
        if ($Cw !== $C) {
            throw new \InvalidArgumentException("conv2d: channel mismatch ({$C} vs {$Cw}).");
        }

        $outH = intdiv($H + 2 * $padding - $kH, $stride) + 1;
        $outW = intdiv($W + 2 * $padding - $kW, $stride) + 1;

        $backend = self::backend();

        if (method_exists($backend, 'conv2dForward')) {
            // ---- Native fast path ----
            $out = $backend->conv2dForward(
                $this->data, $weight->data(), $bias->data(),
                $C, $H, $W, $outC, $kH, $kW, $stride, $padding,
            );
        } else {
            // ---- Pure PHP fallback ----
            $inD = $this->data;
            $wD  = $weight->data();
            $bD  = $bias->data();
            $out = [];
            for ($oc = 0; $oc < $outC; $oc++) {
                for ($oh = 0; $oh < $outH; $oh++) {
                    for ($ow = 0; $ow < $outW; $ow++) {
                        $sum = $bD[$oc];
                        for ($c = 0; $c < $C; $c++) {
                            $inCBase = $c * $H * $W;
                            $wCBase  = $oc * $C * $kH * $kW + $c * $kH * $kW;
                            for ($kh = 0; $kh < $kH; $kh++) {
                                $ih = $oh * $stride - $padding + $kh;
                                if ($ih < 0 || $ih >= $H) continue;
                                for ($kw = 0; $kw < $kW; $kw++) {
                                    $iw = $ow * $stride - $padding + $kw;
                                    if ($iw < 0 || $iw >= $W) continue;
                                    $sum += $inD[$inCBase + $ih * $W + $iw]
                                          * $wD[$wCBase + $kh * $kW + $kw];
                                }
                            }
                        }
                        $out[] = $sum;
                    }
                }
            }
        }

        $result = new self($out, new Shape([$outC, $outH, $outW]), $this->dtype, $this->device);

        if (self::gradEnabled()
            && ($this->requiresGrad || $weight->requiresGrad || $bias->requiresGrad)
        ) {
            self::attachAutograd(
                $result,
                [$this, $weight, $bias],
                'conv2d',
                function (Tensor $g) use (
                    $result, $outC, $outH, $outW, $C, $H, $W, $kH, $kW,
                    $stride, $padding
                ): void {
                    [$input, $weight, $bias] = $result->inputs;
                    $backend = self::backend();

                    if (method_exists($backend, 'conv2dBackward')) {
                        [$gradInput, $gradWeight, $gradBias] = $backend->conv2dBackward(
                            $input->data(), $weight->data(), $g->data(),
                            $C, $H, $W, $outC, $kH, $kW, $stride, $padding,
                        );
                    } else {
                        $inD = $input->data();
                        $wD  = $weight->data();
                        $gd  = $g->data();

                        $gradInput  = array_fill(0, $C * $H * $W, 0.0);
                        $gradWeight = array_fill(0, $outC * $C * $kH * $kW, 0.0);
                        $gradBias   = array_fill(0, $outC, 0.0);

                        for ($oc = 0; $oc < $outC; $oc++) {
                            for ($oh = 0; $oh < $outH; $oh++) {
                                for ($ow = 0; $ow < $outW; $ow++) {
                                    $gVal = $gd[$oc * $outH * $outW + $oh * $outW + $ow];
                                    $gradBias[$oc] += $gVal;

                                    for ($c = 0; $c < $C; $c++) {
                                        $inCBase = $c * $H * $W;
                                        $wCBase  = $oc * $C * $kH * $kW + $c * $kH * $kW;
                                        for ($kh = 0; $kh < $kH; $kh++) {
                                            $ih = $oh * $stride - $padding + $kh;
                                            if ($ih < 0 || $ih >= $H) continue;
                                            for ($kw = 0; $kw < $kW; $kw++) {
                                                $iw = $ow * $stride - $padding + $kw;
                                                if ($iw < 0 || $iw >= $W) continue;

                                                $iIdx = $inCBase + $ih * $W + $iw;
                                                $wIdx = $wCBase + $kh * $kW + $kw;

                                                $gradInput[$iIdx]  += $gVal * $wD[$wIdx];
                                                $gradWeight[$wIdx] += $gVal * $inD[$iIdx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if ($input->requiresGrad) {
                        $input->accumulateGradPublic(new self($gradInput, $input->shape()));
                    }
                    if ($weight->requiresGrad) {
                        $weight->accumulateGradPublic(new self($gradWeight, $weight->shape()));
                    }
                    if ($bias->requiresGrad) {
                        $bias->accumulateGradPublic(new self($gradBias, $bias->shape()));
                    }
                },
            );
        }

        return $result;
    }

    /**
     * Max pooling. Uses the native kernel when available.
     */
    public function maxPool2d(int $kernelSize, ?int $stride = null): self
    {
        $stride ??= $kernelSize;

        $dims = $this->shape->dims();
        if (count($dims) !== 3) {
            throw new \RuntimeException("maxPool2d input must be [C, H, W].");
        }
        [$C, $H, $W] = $dims;
        $outH = intdiv($H - $kernelSize, $stride) + 1;
        $outW = intdiv($W - $kernelSize, $stride) + 1;

        $backend = self::backend();

        if (method_exists($backend, 'maxPool2dForward')) {
            [$out, $argmax] = $backend->maxPool2dForward(
                $this->data, $C, $H, $W, $kernelSize, $kernelSize, $stride
            );
        } else {
            $inD = $this->data;
            $out = [];
            $argmax = [];

            for ($c = 0; $c < $C; $c++) {
                $inCBase = $c * $H * $W;
                for ($oh = 0; $oh < $outH; $oh++) {
                    for ($ow = 0; $ow < $outW; $ow++) {
                        $best = -INF;
                        $bestIdx = 0;
                        for ($kh = 0; $kh < $kernelSize; $kh++) {
                            for ($kw = 0; $kw < $kernelSize; $kw++) {
                                $ih = $oh * $stride + $kh;
                                $iw = $ow * $stride + $kw;
                                $idx = $inCBase + $ih * $W + $iw;
                                $v = $inD[$idx];
                                if ($v > $best) { $best = $v; $bestIdx = $idx; }
                            }
                        }
                        $out[] = $best;
                        $argmax[] = $bestIdx;
                    }
                }
            }
        }

        $result = new self($out, new Shape([$C, $outH, $outW]), $this->dtype, $this->device);

        if (self::gradEnabled() && $this->requiresGrad) {
            self::attachAutograd(
                $result,
                [$this],
                'max_pool2d',
                function (Tensor $g) use ($result, $argmax, $C, $H, $W, $outH, $outW): void {
                    $input = $result->inputs[0];
                    $backend = self::backend();

                    if (method_exists($backend, 'maxPool2dBackward')) {
                        $gradInput = $backend->maxPool2dBackward(
                            $g->data(), $argmax, $C, $H, $W, $outH, $outW
                        );
                    } else {
                        $gradInput = array_fill(0, $C * $H * $W, 0.0);
                        $gd = $g->data();
                        foreach ($argmax as $i => $idx) {
                            $gradInput[$idx] += $gd[$i];
                        }
                    }
                    $input->accumulateGradPublic(new self($gradInput, $input->shape()));
                },
            );
        }

        return $result;
    }

    public function flatten(): self
    {
        $dims = $this->shape->dims();
        $total = array_product($dims);

        $result = new self($this->data, new Shape([1, $total]), $this->dtype, $this->device);

        if (self::gradEnabled() && $this->requiresGrad) {
            $origShape = $dims;
            self::attachAutograd(
                $result,
                [$this],
                'flatten',
                function (Tensor $g) use ($result, $origShape): void {
                    $input = $result->inputs[0];
                    $input->accumulateGradPublic(
                        new self($g->data(), new Shape($origShape))
                    );
                },
            );
        }

        return $result;
    }

    // ==================================================================
    // Differentiable ops with broadcasting
    // ==================================================================

    private function binaryOp(
        Tensor $other,
        string $opName,
        callable $forward,
        callable $backwardA,
        callable $backwardB,
    ): Tensor {
        $aDims = $this->shape()->dims();
        $bDims = $other->shape()->dims();
        $targetDims = self::broadcastShapes($aDims, $bDims);

        $aB = self::broadcastTo($this, $targetDims);
        $bB = self::broadcastTo($other, $targetDims);

        $result = $forward($aB, $bB);

        if (self::gradEnabled() && ($this->requiresGrad || $other->requiresGrad)) {
            $result->requiresGrad = true;
            $result->op = $opName;
            $result->inputs = [$this, $other];
            $result->backwardFn = function (Tensor $g) use (
                $result, $aDims, $bDims, $backwardA, $backwardB, $aB, $bB
            ): void {
                [$a, $b] = $result->inputs;
                if ($a->requiresGrad) {
                    $ga = $backwardA($g, $aB, $bB, $result);
                    $a->accumulateGrad(self::reduceGrad($ga, $aDims));
                }
                if ($b->requiresGrad) {
                    $gb = $backwardB($g, $aB, $bB, $result);
                    $b->accumulateGrad(self::reduceGrad($gb, $bDims));
                }
            };
        }
        return $result;
    }

    public function add(Tensor $other): Tensor
    {
        return $this->binaryOp(
            $other, 'add',
            fn($A, $B) => self::backend()->add($A, $B),
            fn($g) => $g,
            fn($g) => $g,
        );
    }

    public function sub(Tensor $other): Tensor
    {
        return $this->binaryOp(
            $other, 'sub',
            fn($A, $B) => self::backend()->subtract($A, $B),
            fn($g) => $g,
            fn($g, $a, $b, $r) => self::backend()->multiply(
                $g, self::full($g->shape()->dims(), -1.0)
            ),
        );
    }

    public function mul(Tensor $other): Tensor
    {
        return $this->binaryOp(
            $other, 'mul',
            fn($A, $B) => self::backend()->multiply($A, $B),
            fn($g, $a, $b) => self::backend()->multiply($g, $b),
            fn($g, $a, $b) => self::backend()->multiply($g, $a),
        );
    }

    public function div(Tensor $other): Tensor
    {
        return $this->binaryOp(
            $other, 'div',
            fn($A, $B) => self::backend()->divide($A, $B),
            fn($g, $a, $b) => self::backend()->divide($g, $b),
            function ($g, $a, $b) {
                $bSq    = self::backend()->multiply($b, $b);
                $inner  = self::backend()->divide($a, $bSq);
                $neg    = self::backend()->multiply($inner, self::full($inner->shape()->dims(), -1.0));
                return self::backend()->multiply($g, $neg);
            },
        );
    }

    public function matmul(Tensor $other): Tensor
    {
        $result = self::backend()->matmul($this, $other);
        if (self::gradEnabled() && ($this->requiresGrad || $other->requiresGrad)) {
            $result->requiresGrad = true;
            $result->op = 'matmul';
            $result->inputs = [$this, $other];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                [$a, $b] = $result->inputs;
                if ($a->requiresGrad) {
                    $a->accumulateGrad(self::backend()->matmul($g, $b->transpose()));
                }
                if ($b->requiresGrad) {
                    $b->accumulateGrad(self::backend()->matmul($a->transpose(), $g));
                }
            };
        }
        return $result;
    }

    public function transpose(): Tensor
    {
        $dims = $this->shape()->dims();
        if (count($dims) !== 2) throw new \RuntimeException("transpose requires 2D.");
        [$m, $n] = $dims;
        $d = $this->data(); $out = [];
        for ($j = 0; $j < $n; $j++)
            for ($i = 0; $i < $m; $i++)
                $out[] = $d[$i * $n + $j];
        $result = new Tensor($out, new Shape([$n, $m]));
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true;
            $result->op = 'transpose';
            $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $a->accumulateGrad($g->transpose());
            };
        }
        return $result;
    }

    public function sum(): Tensor
    {
        $result = self::backend()->sum($this);
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true; $result->op = 'sum'; $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $a->accumulateGrad(self::full($a->shape()->dims(), (float) $g->item()));
            };
        }
        return $result;
    }

    public function mean(): Tensor
    {
        $result = self::backend()->mean($this);
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true; $result->op = 'mean'; $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $n = max($a->shape()->size(), 1);
                $a->accumulateGrad(self::full($a->shape()->dims(), (float) $g->item() / $n));
            };
        }
        return $result;
    }

    public function relu(): Tensor
    {
        $result = self::backend()->relu($this);
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true; $result->op = 'relu'; $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $mask = [];
                foreach ($a->data() as $v) $mask[] = $v > 0 ? 1.0 : 0.0;
                $maskT = new Tensor($mask, $a->shape());
                $a->accumulateGrad(self::backend()->multiply($g, $maskT));
            };
        }
        return $result;
    }

    public function exp(): Tensor
    {
        $result = self::backend()->exp($this);
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true; $result->op = 'exp'; $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $y = self::backend()->exp($a);
                $a->accumulateGrad(self::backend()->multiply($g, $y));
            };
        }
        return $result;
    }

    public function log(): Tensor
    {
        $result = self::backend()->log($this);
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true; $result->op = 'log'; $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $inv = self::backend()->divide(self::ones($a->shape()->dims()), $a);
                $a->accumulateGrad(self::backend()->multiply($g, $inv));
            };
        }
        return $result;
    }

    public function softmax(): Tensor
    {
        $dims = $this->shape()->dims();
        if (count($dims) !== 2) throw new \RuntimeException("softmax requires 2D input.");
        [$rows, $cols] = $dims;
        $d = $this->data();
        $y = [];

        for ($i = 0; $i < $rows; $i++) {
            $row = array_slice($d, $i * $cols, $cols);
            $max = max($row);
            $sum = 0.0;
            $exps = [];
            foreach ($row as $v) { $e = exp($v - $max); $exps[] = $e; $sum += $e; }
            foreach ($exps as $e) $y[] = $e / $sum;
        }

        $result = new Tensor($y, $this->shape());
        if (self::gradEnabled() && $this->requiresGrad) {
            $result->requiresGrad = true; $result->op = 'softmax'; $result->inputs = [$this];
            $result->backwardFn = function (Tensor $g) use ($result): void {
                $a = $result->inputs[0];
                $yData = $result->data();
                $gData = $g->data();
                [$rows, $cols] = $result->shape()->dims();
                $dx = [];
                for ($i = 0; $i < $rows; $i++) {
                    $dot = 0.0;
                    for ($j = 0; $j < $cols; $j++) $dot += $gData[$i*$cols+$j] * $yData[$i*$cols+$j];
                    for ($j = 0; $j < $cols; $j++) {
                        $dx[] = $yData[$i*$cols+$j] * ($gData[$i*$cols+$j] - $dot);
                    }
                }
                $a->accumulateGrad(new Tensor($dx, $result->shape()));
            };
        }
        return $result;
    }

    public function __toString(): string
    {
        $g = $this->requiresGrad ? ', requiresGrad=true' : '';
        return "Tensor(shape={$this->shape}, dtype={$this->dtype->value}, device={$this->device}{$g})";
    }
}