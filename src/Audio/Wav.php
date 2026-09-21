<?php

declare(strict_types=1);

namespace ZillaPHP\Audio;

use ZillaPHP\Tensor\Shape\Shape;
use ZillaPHP\Tensor\Tensor;

/**
 * Minimal WAV loader for 16-bit PCM (mono or stereo).
 * Produces a float32 tensor of shape [channels, samples] with values
 * normalized to [-1.0, 1.0].
 *
 * Not a general-purpose audio library — enough to load Speech Commands,
 * LibriSpeech, and most common datasets. Compressed formats (MP3, OGG,
 * FLAC) are not supported here; FFmpeg integration is a later phase.
 */
final class Wav
{
    public readonly int $sampleRate;
    public readonly int $channels;
    public readonly int $numSamples;
    /** @var float[] interleaved [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...] */
    public readonly array $data;

    private function __construct(
        int $sampleRate,
        int $channels,
        int $numSamples,
        array $data,
    ) {
        $this->sampleRate  = $sampleRate;
        $this->channels    = $channels;
        $this->numSamples  = $numSamples;
        $this->data        = $data;
    }

    public static function fromFile(string $path): self
    {
        if (!is_file($path)) {
            throw new \RuntimeException("WAV file not found: {$path}");
        }

        $raw = file_get_contents($path);
        if ($raw === false || strlen($raw) < 44) {
            throw new \RuntimeException("File too small to be a WAV: {$path}");
        }

        // RIFF header
        if (substr($raw, 0, 4) !== 'RIFF' || substr($raw, 8, 4) !== 'WAVE') {
            throw new \RuntimeException("Not a RIFF/WAVE file: {$path}");
        }

        // Walk chunks (fmt, data, possibly others)
        $pos = 12;
        $len = strlen($raw);

        $fmtChunk  = null;
        $dataChunk = null;
        $dataChunkLen = 0;

        while ($pos + 8 <= $len) {
            $chunkId   = substr($raw, $pos, 4);
            $chunkSize = unpack('V', substr($raw, $pos + 4, 4))[1];
            $chunkBody = $pos + 8;

            if ($chunkId === 'fmt ') {
                $fmtChunk = substr($raw, $chunkBody, $chunkSize);
            } elseif ($chunkId === 'data') {
                $dataChunk    = $chunkBody;
                $dataChunkLen = $chunkSize;
            }

            // Chunks are word-aligned (padded to even byte count)
            $pos = $chunkBody + $chunkSize;
            if ($chunkSize % 2 !== 0) $pos++;
        }

        if ($fmtChunk === null)  throw new \RuntimeException("Missing 'fmt ' chunk.");
        if ($dataChunk === null) throw new \RuntimeException("Missing 'data' chunk.");

        // fmt chunk layout (16 bytes minimum for PCM):
        //   audioFormat(u16) | numChannels(u16) | sampleRate(u32) |
        //   byteRate(u32) | blockAlign(u16) | bitsPerSample(u16)
        $fmt         = unpack(
            'vaudioFormat/vnumChannels/VsampleRate/VbyteRate/vblockAlign/vbitsPerSample',
            $fmtChunk
        );
        $audioFormat = $fmt['audioFormat'];
        $numChannels = $fmt['numChannels'];
        $sampleRate  = $fmt['sampleRate'];
        $bitsPerSamp = $fmt['bitsPerSample'];

        if ($audioFormat !== 1) {
            throw new \RuntimeException("Only uncompressed PCM supported (got format {$audioFormat}).");
        }
        if ($bitsPerSamp !== 16) {
            throw new \RuntimeException("Only 16-bit PCM supported (got {$bitsPerSamp}-bit).");
        }
        if ($numChannels < 1 || $numChannels > 2) {
            throw new \RuntimeException("Only mono or stereo supported (got {$numChannels} channels).");
        }

        $bytesPerSample = 2;
        $frameCount     = intdiv($dataChunkLen, $bytesPerSample * $numChannels);

        // Decode interleaved int16 → float32 in [-1, 1]
        $floats  = [];
        $bytes   = substr($raw, $dataChunk, $frameCount * $numChannels * $bytesPerSample);
        $samples = unpack('v*', $bytes);   // 'v' = unsigned 16-bit LE

        foreach ($samples as $u16) {
            $s16 = $u16 >= 0x8000 ? $u16 - 0x10000 : $u16;
            $floats[] = $s16 / 32768.0;
        }

        return new self($sampleRate, $numChannels, $frameCount, $floats);
    }

    /**
     * @return Tensor  shape [channels, numSamples]
     */
    public function toTensor(): Tensor
    {
        if ($this->channels === 1) {
            return new Tensor($this->data, new Shape([1, $this->numSamples]));
        }

        // De-interleave for [C, T] layout
        $ch0 = [];
        $ch1 = [];
        for ($i = 0; $i < $this->numSamples; $i++) {
            $ch0[] = $this->data[$i * 2];
            $ch1[] = $this->data[$i * 2 + 1];
        }
        return new Tensor(
            array_merge($ch0, $ch1),
            new Shape([2, $this->numSamples])
        );
    }

    public function durationSeconds(): float
    {
        return $this->numSamples / $this->sampleRate;
    }

    public function summary(): string
    {
        return sprintf(
            "Wav(%d Hz, %d ch, %d samples, %.2f s)",
            $this->sampleRate,
            $this->channels,
            $this->numSamples,
            $this->durationSeconds()
        );
    }
}