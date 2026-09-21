<?php

declare(strict_types=1);

namespace ZillaPHP\Audio;

use ZillaPHP\Hardware\CPU\NativeCpuBackend;
use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

/**
 * Full audio → log-mel spectrogram pipeline.
 *
 *   Wav → waveform → STFT (native) → mel (native matmul) → log
 *
 * Output tensor shape: [1, nMels, nFrames]
 *   - the leading 1 is the "channel" dim so the tensor can be fed
 *     directly into Conv2D (which expects [C, H, W]).
 */
final class MelSpectrogram
{
    public function __construct(
        private NativeCpuBackend $backend,
        private int $sampleRate,
        private int $fftSize = 512,
        private int $hopSize = 160,
        private int $nMels   = 64,
        private float $fMin  = 0.0,
        private ?float $fMax = null,
        private float $logEps = 1e-6,
    ) {}

    private ?array $window = null;
    private ?MelFilterbank $melBank = null;

    public function nMels(): int { return $this->nMels; }
    public function fftSize(): int { return $this->fftSize; }
    public function hopSize(): int { return $this->hopSize; }

    private function ensureAssets(): void
    {
        if ($this->window === null) {
            $this->window = $this->backend->hannWindow($this->fftSize);
        }
        if ($this->melBank === null) {
            $this->melBank = new MelFilterbank(
                sampleRate: $this->sampleRate,
                fftSize:    $this->fftSize,
                nMels:      $this->nMels,
                fMin:       $this->fMin,
                fMax:       $this->fMax,
            );
        }
    }

    /**
     * @param Wav $wav
     * @return Tensor   shape [1, nMels, nFrames]
     */
    public function fromWav(Wav $wav): Tensor
    {
        if ($wav->channels !== 1) {
            throw new \RuntimeException(
                "MelSpectrogram expects mono audio; got {$wav->channels} channels."
            );
        }
        if ($wav->sampleRate !== $this->sampleRate) {
            throw new \RuntimeException(
                "Sample rate mismatch: WAV={$wav->sampleRate}, " .
                "expected={$this->sampleRate}. Resampling is a later phase."
            );
        }

        $this->ensureAssets();

        // 1. STFT magnitude [nBins, nFrames] (native)
        [$mag, $nBins, $nFrames] = $this->backend->stftMagnitude(
            $wav->data, $this->window, $this->fftSize, $this->hopSize
        );

        // 2. Transpose to [nFrames, nBins] via a fresh tensor
        $magT = [];
        for ($f = 0; $f < $nFrames; $f++) {
            for ($b = 0; $b < $nBins; $b++) {
                $magT[] = $mag[$b * $nFrames + $f];
            }
        }
        $magTensor = new Tensor($magT, new Shape([$nFrames, $nBins]));

        // 3. Mel matrix multiply (native)
        // melBankFlat is [nMels, nBins] — we need [nBins, nMels] for matmul
        $melFlat = $this->melBank->flattened();    // row-major [nMels, nBins]
        $melTransposed = [];
        for ($b = 0; $b < $nBins; $b++) {
            for ($m = 0; $m < $this->nMels; $m++) {
                $melTransposed[] = $melFlat[$m * $nBins + $b];
            }
        }
        $melTensor = new Tensor($melTransposed, new Shape([$nBins, $this->nMels]));

        $melOut = $magTensor->matmul($melTensor);   // [nFrames, nMels]

        // 4. Transpose back to [nMels, nFrames]
        $melData = $melOut->data();
        $melFinal = [];
        for ($m = 0; $m < $this->nMels; $m++) {
            for ($f = 0; $f < $nFrames; $f++) {
                $melFinal[] = $melData[$f * $this->nMels + $m];
            }
        }

        // 5. log(x + eps)
        foreach ($melFinal as $i => $v) {
            $melFinal[$i] = log($v + $this->logEps);
        }

        // 6. Return as [1, nMels, nFrames] tensor
        return new Tensor($melFinal, new Shape([1, $this->nMels, $nFrames]));
    }
}