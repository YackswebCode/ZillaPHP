<?php

declare(strict_types=1);

namespace ZillaPHP\Serialization;

/**
 * SafeTensors serializer.
 *
 * Format: https://github.com/huggingface/safetensors
 *
 *   [8 bytes]  header_size (u64, little-endian)
 *   [H bytes]  JSON header (UTF-8), padded with spaces to 8-byte alignment
 *   [D bytes]  tensor data, concatenated in the order declared in the header
 *
 * Only F32 is written by default. F16 and BF16 can be added later.
 */
final class SafeTensorsSerializer implements Serializer
{
    /** Bytes per element for each dtype we can serialize. */
    private const DTYPE_SIZE = [
        'F32'  => 4,
        'F16'  => 2,
        'BF16' => 2,
        'I32'  => 4,
        'I64'  => 8,
    ];

    /** Chunk size for pack() — avoids the O(n²) spread cost on huge tensors. */
    private const PACK_CHUNK = 4096;

    public function save(array $state, string $path): void
    {
        $layers   = $state['layers']   ?? [];
        $metadata = $state['metadata'] ?? null;

        $header   = [];
        $dataBlob = '';
        $offset   = 0;

        foreach ($layers as $name => $tensor) {
            $shape = $tensor['shape'] ?? [];
            $flat  = $tensor['data']  ?? [];

            $dtype = 'F32';
            $n     = count($flat);
            $bytes = $n * self::DTYPE_SIZE[$dtype];

            $header[$name] = [
                'dtype'        => $dtype,
                'shape'        => array_map('intval', $shape),
                'data_offsets' => [$offset, $offset + $bytes],
            ];

            // Pack floats in chunks to avoid spreading a huge array
            foreach (array_chunk($flat, self::PACK_CHUNK) as $chunk) {
                $dataBlob .= pack('f*', ...array_map('floatval', $chunk));
            }

            $offset += $bytes;
        }

        // Metadata is stored in the header under the reserved __metadata__ key
        if ($metadata !== null) {
            $header['__metadata__'] = $metadata;
        }

        $headerJson  = json_encode($header, JSON_UNESCAPED_SLASHES);
        $headerBytes = strlen($headerJson);

        // Pad to 8-byte boundary
        if ($headerBytes % 8 !== 0) {
            $pad = 8 - ($headerBytes % 8);
            $headerJson .= str_repeat(' ', $pad);
            $headerBytes += $pad;
        }

        $dir = dirname($path);
        if (!is_dir($dir)) mkdir($dir, 0777, true);

        $fh = fopen($path, 'wb');
        if (!$fh) {
            throw new \RuntimeException("Cannot open {$path} for writing.");
        }

        fwrite($fh, pack('P', $headerBytes));   // u64 LE
        fwrite($fh, $headerJson);
        fwrite($fh, $dataBlob);
        fclose($fh);
    }

    public function load(string $path): array
    {
        if (!is_file($path)) {
            throw new \RuntimeException("Checkpoint not found: {$path}");
        }

        $raw = file_get_contents($path);
        if ($raw === false || strlen($raw) < 8) {
            throw new \RuntimeException("File too small to be a SafeTensors file.");
        }

        $headerSize = unpack('P', substr($raw, 0, 8))[1];

        if (strlen($raw) < 8 + $headerSize) {
            throw new \RuntimeException("Truncated SafeTensors file.");
        }

        $headerJson = substr($raw, 8, $headerSize);
        $dataStart  = 8 + $headerSize;

        $header = json_decode($headerJson, true);
        if (!is_array($header)) {
            throw new \RuntimeException("Invalid SafeTensors header (not JSON).");
        }

        $layers   = [];
        $metadata = null;

        foreach ($header as $name => $info) {
            if ($name === '__metadata__') {
                $metadata = $info;
                continue;
            }

            if (!isset($info['dtype'], $info['shape'], $info['data_offsets'])) {
                throw new \RuntimeException("Malformed entry '{$name}' in header.");
            }

            $dtype = $info['dtype'];
            if (!isset(self::DTYPE_SIZE[$dtype])) {
                throw new \RuntimeException("Unsupported dtype '{$dtype}' for '{$name}'.");
            }
            if ($dtype !== 'F32') {
                throw new \RuntimeException(
                    "Only F32 is currently supported on load (got '{$dtype}')."
                );
            }

            [$start, $end] = $info['data_offsets'];
            $bytes = substr($raw, $dataStart + $start, $end - $start);

            $flat = array_values(unpack('f*', $bytes));

            $layers[$name] = [
                'shape' => $info['shape'],
                'data'  => $flat,
            ];
        }

        $state = ['layers' => $layers];
        if ($metadata !== null) {
            $state['metadata'] = $metadata;
        }

        return $state;
    }
}