<?php

declare(strict_types=1);

namespace ZillaPHP\Data;

use ZillaPHP\Tensor\Tensor;

/**
 * Loads raw MNIST IDX files into ZillaPHP tensors.
 *
 * Each sample is:
 *   input:  Tensor [1, 784]  — pixel values normalized to [0, 1]
 *   target: [int]            — digit class 0..9
 */
final class MnistDataset implements Dataset
{
    /** @var string */
    private string $imagesRaw;
    /** @var string */
    private string $labelsRaw;
    private int $count;
    private int $rows;
    private int $cols;

    public function __construct(string $imagesPath, string $labelsPath)
    {
        if (!is_file($imagesPath) || !is_file($labelsPath)) {
            throw new \RuntimeException(
                "MNIST files not found. Run: php scripts/download_mnist.php"
            );
        }

        $this->imagesRaw = file_get_contents($imagesPath);
        $this->labelsRaw = file_get_contents($labelsPath);

        $header = unpack('Nmagic/Ncount/Nrows/Ncols', substr($this->imagesRaw, 0, 16));
        if ($header['magic'] !== 2051) {
            throw new \RuntimeException("Bad images magic number.");
        }
        $this->count = $header['count'];
        $this->rows  = $header['rows'];
        $this->cols  = $header['cols'];

        $labelHeader = unpack('Nmagic/Ncount', substr($this->labelsRaw, 0, 8));
        if ($labelHeader['magic'] !== 2049 || $labelHeader['count'] !== $this->count) {
            throw new \RuntimeException("Label file mismatch.");
        }
    }

    public function size(): int
    {
        return $this->count;
    }

    public function rows(): int { return $this->rows; }
    public function cols(): int { return $this->cols; }

    /** @return array{0: Tensor, 1: int[]} */
    public function get(int $index): array
    {
        if ($index < 0 || $index >= $this->count) {
            throw new \OutOfRangeException("Index {$index} out of range.");
        }

        $pixels = 28 * 28;
        $offset = 16 + $index * $pixels;
        $raw    = substr($this->imagesRaw, $offset, $pixels);

        // Normalize to [0, 1]
        $values = [];
        for ($i = 0; $i < $pixels; $i++) {
            $values[] = ord($raw[$i]) / 255.0;
        }

        $label = ord($this->labelsRaw[8 + $index]);

        return [
            Tensor::fromArray([$values]),   // [1, 784]
            [$label],
        ];
    }
}