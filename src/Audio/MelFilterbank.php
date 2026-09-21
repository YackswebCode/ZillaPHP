<?php

declare(strict_types=1);

namespace ZillaPHP\Audio;

/**
 * Mel filterbank construction (HTK / Slaney-style formulas).
 *
 * Given:
 *   - sampleRate
 *   - fftSize      → nBins = fftSize/2 + 1
 *   - nMels
 *   - fMin, fMax
 *
 * Produces a dense [nMels, nBins] matrix of triangular filters.
 * Multiply [nFrames, nBins] @ [nBins, nMels]ᵀ → mel spectrogram.
 */
final class MelFilterbank
{
    public readonly array $matrix;  // [nMels][nBins]  — dense, row-major
    public readonly int $nMels;
    public readonly int $nBins;

    public function __construct(
        int $sampleRate,
        int $fftSize,
        int $nMels = 64,
        float $fMin = 0.0,
        ?float $fMax = null,
    ) {
        if ($fftSize < 2 || ($fftSize & ($fftSize - 1)) !== 0) {
            throw new \InvalidArgumentException("fftSize must be a power of two.");
        }
        if ($nMels < 1) {
            throw new \InvalidArgumentException("nMels must be >= 1.");
        }

        $fMax ??= $sampleRate / 2.0;
        $this->nMels = $nMels;
        $this->nBins = intdiv($fftSize, 2) + 1;

        // 1. Mel-spaced boundary points: nMels + 2 of them
        $melMin = self::hzToMel($fMin);
        $melMax = self::hzToMel($fMax);
        $melPoints = [];
        for ($i = 0; $i < $nMels + 2; $i++) {
            $melPoints[] = $melMin + ($melMax - $melMin) * $i / ($nMels + 1);
        }

        // 2. Convert to Hz, then to FFT bin indices
        $binPoints = [];
        for ($i = 0; $i < $nMels + 2; $i++) {
            $hz = self::melToHz($melPoints[$i]);
            $binPoints[] = ($fftSize + 1) * $hz / $sampleRate;
        }

        // 3. Build triangular filters
        $matrix = [];
        for ($m = 0; $m < $nMels; $m++) {
            $left   = $binPoints[$m];
            $center = $binPoints[$m + 1];
            $right  = $binPoints[$m + 2];

            $row = array_fill(0, $this->nBins, 0.0);

            // Rising edge: [left, center)
            for ($b = (int) ceil($left); $b < (int) ceil($center) && $b < $this->nBins; $b++) {
                if ($b <= $left) continue;
                if ($center === $left) continue;
                $w = ($b - $left) / ($center - $left);
                $row[$b] = $w;
            }
            // Falling edge: [center, right)
            for ($b = (int) floor($center); $b <= (int) floor($right) && $b < $this->nBins; $b++) {
                if ($b >= $right) break;
                if ($right === $center) continue;
                $w = ($right - $b) / ($right - $center);
                // If we already set a rising-edge value, use max (triangle peak)
                if ($w > $row[$b]) $row[$b] = $w;
            }

            // Area-normalize (Slaney-style) so each filter sums to ~1
            $sum = array_sum($row);
            if ($sum > 1e-12) {
                foreach ($row as $i => $v) $row[$i] = $v / $sum;
            }

            $matrix[] = $row;
        }

        $this->matrix = $matrix;
    }

    public static function hzToMel(float $hz): float
    {
        return 2595.0 * log10(1.0 + $hz / 700.0);
    }

    public static function melToHz(float $mel): float
    {
        return 700.0 * (pow(10.0, $mel / 2595.0) - 1.0);
    }

    /**
     * Flattened matrix [nMels * nBins] for direct use with the backend.
     *
     * @return float[]
     */
    public function flattened(): array
    {
        $out = [];
        foreach ($this->matrix as $row) {
            foreach ($row as $v) $out[] = $v;
        }
        return $out;
    }
}