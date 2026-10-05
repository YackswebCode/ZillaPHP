<?php

declare(strict_types=1);

namespace ZillaPHP\Data;

use ZillaPHP\Audio\MelSpectrogram;
use ZillaPHP\Hardware\CPU\NativeCpuBackend;
use ZillaPHP\Tensor\Tensor;

/**
 * Google Speech Commands v2 dataset preprocessed to log-mel spectrograms.
 *
 * Each sample becomes a [1, 64, 101] tensor (1 second at 16 kHz).
 */
final class SpeechCommandsDataset implements Dataset
{
    /** @var array<int, array{0: string, 1: int}> */
    private array $items;

    /** @var string[] */
    private array $classes;

    private NativeCpuBackend $backend;

    /** @var float[]|null  Hann window, cached */
    private ?array $window = null;

    /** @var array{nBins:int, nFrames:int, melFlat:float[]}|null */
    private ?array $melAssets = null;

    public function __construct(
        string $root,
        string $split = 'train',
        int $maxPerClass = 0,
        ?NativeCpuBackend $backend = null,
    ) {
        if (!is_dir($root)) {
            throw new \RuntimeException("Speech Commands not found at {$root}.");
        }

        $this->backend = $backend ?? new NativeCpuBackend();

        // ---- Classes = subdirs with WAVs ----
        $classes = [];
        foreach (scandir($root) as $entry) {
            if ($entry === '.' || $entry === '..') continue;
            $sub = "{$root}/{$entry}";
            if (is_dir($sub) && count(glob("{$sub}/*.wav")) > 0) $classes[] = $entry;
        }
        sort($classes);
        $this->classes = $classes;

        // ---- Split lists (official) ----
        $valList  = $this->loadSplitList("{$root}/validation_list.txt");
        $testList = $this->loadSplitList("{$root}/testing_list.txt");

        $counts = [];
        $items  = [];

        foreach ($classes as $classIdx => $class) {
            $wavs = glob("{$root}/{$class}/*.wav");
            sort($wavs);
            foreach ($wavs as $p) {
                $rel = $class . '/' . basename($p);
                $itemSplit = 'train';
                if (in_array($rel, $testList, true))       $itemSplit = 'test';
                elseif (in_array($rel, $valList, true))    $itemSplit = 'val';

                if ($itemSplit !== $split) continue;
                if ($maxPerClass > 0 && ($counts[$class] ?? 0) >= $maxPerClass) continue;

                $items[] = [$p, $classIdx];
                $counts[$class] = ($counts[$class] ?? 0) + 1;
            }
        }

        $this->items = $items;
    }

    public function classes(): array { return $this->classes; }
    public function size(): int { return count($this->items); }

    /** @return array{0: Tensor, 1: int} */
    public function get(int $index): array
    {
        if (!isset($this->items[$index])) {
            throw new \OutOfRangeException("Index {$index} out of range.");
        }

        [$path, $label] = $this->items[$index];
        $mel = $this->computeLogMel($path);
        return [new Tensor($mel, new \ZillaPHP\Tensor\Shape\Shape([1, 64, 101])), $label];
    }

    /** @return float[]  flat [64 * 101] log-mel */
    private function computeLogMel(string $wavPath): array
    {
        // ---- Decode WAV manually (we can't use Wav class here — private ctor) ----
        $raw = file_get_contents($wavPath);
        if ($raw === false) throw new \RuntimeException("Cannot read {$wavPath}");

        // Skip 44-byte header (standard PCM WAV)
        $samples = [];
        $n = (strlen($raw) - 44) / 2;
        for ($i = 0; $i < $n; $i++) {
            $u16 = unpack('v', substr($raw, 44 + $i * 2, 2))[1];
            $s16 = $u16 >= 0x8000 ? $u16 - 0x10000 : $u16;
            $samples[] = $s16 / 32768.0;
        }

        // Pad or trim to 16000 samples
        $target = 16000;
        $n = count($samples);
        if ($n < $target) {
            $samples = array_merge($samples, array_fill(0, $target - $n, 0.0));
        } elseif ($n > $target) {
            $samples = array_slice($samples, 0, $target);
        }

        // ---- Hann window + STFT ----
        if ($this->window === null) {
            $this->window = $this->backend->hannWindow(512);
        }

        [$mag, $nBins, $nFrames] = $this->backend->stftMagnitude(
            $samples, $this->window, 512, 160
        );

        // Note: 16000 samples with 512-FFT, 160-hop → (16000-512)/160 + 1 = 97 frames.
        // We need 101 frames for the padded clip; zero-pad the missing 4 frames.

        // ---- Mel filterbank ----
        if ($this->melAssets === null) {
            $bank = new \ZillaPHP\Audio\MelFilterbank(16000, 512, 64);
            $melFlat = $bank->flattened();  // [64, 257]
            $this->melAssets = [
                'nBins'   => $nBins,
                'nFrames' => 101,
                'melFlat' => $melFlat,
            ];
        }
        $melFlat = $this->melAssets['melFlat'];

        // For each of 64 mel bands, apply to each of nFrames frames
        // mag layout: mag[bin * nFrames + frame]
        $melOut = array_fill(0, 64 * 101, -10.0);  // -10 = log(1e-6), silence floor
        for ($m = 0; $m < 64; $m++) {
            for ($f = 0; $f < $nFrames; $f++) {
                $sum = 0.0;
                for ($b = 0; $b < $nBins; $b++) {
                    $w = $melFlat[$m * 257 + $b];
                    if ($w == 0.0) continue;
                    $sum += $w * $mag[$b * $nFrames + $f];
                }
                $melOut[$m * 101 + $f] = log($sum + 1e-6);
            }
            // Frames 97..100 → leave at floor value (-10)
        }

        return $melOut;
    }

    /** @return string[] */
    private function loadSplitList(string $path): array
    {
        if (!is_file($path)) return [];
        $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        return $lines === false ? [] : $lines;
    }
}