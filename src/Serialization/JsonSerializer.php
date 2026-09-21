<?php

declare(strict_types=1);

namespace ZillaPHP\Serialization;

/**
 * Simple, portable, human-inspectable JSON serialization.
 * SafeTensors / binary formats come later.
 */
final class JsonSerializer implements Serializer
{
    public function save(array $state, string $path): void
    {
        $dir = dirname($path);
        if (!is_dir($dir)) mkdir($dir, 0777, true);

        $json = json_encode(
            [
                'format'    => 'zilla-json',
                'version'   => 1,
                'created_at' => date('c'),
                'state'     => $state,
            ],
            JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES
        );

        if ($json === false) {
            throw new \RuntimeException("Failed to encode model state.");
        }

        if (file_put_contents($path, $json) === false) {
            throw new \RuntimeException("Failed to write to {$path}");
        }
    }

    public function load(string $path): array
    {
        if (!is_file($path)) {
            throw new \RuntimeException("Checkpoint not found: {$path}");
        }

        $raw = file_get_contents($path);
        $data = json_decode($raw, true);

        if (!is_array($data) || ($data['format'] ?? null) !== 'zilla-json') {
            throw new \RuntimeException("Not a ZillaPHP checkpoint: {$path}");
        }

        return $data['state'];
    }
}