<?php

declare(strict_types=1);

namespace ZillaPHP\Serialization;

use ZillaPHP\NN\Module;
use ZillaPHP\Tensor\Tensor;

/**
 * Serializes ZillaPHP models to disk.
 *
 * The serializer backend (JSON, SafeTensors, ...) is pluggable.
 * The on-disk structure is backend-agnostic:
 *
 *     [
 *         'layers' => [
 *             'param_0' => ['shape' => [...], 'data' => [...]],
 *             'param_1' => ['shape' => [...], 'data' => [...]],
 *             ...
 *         ],
 *         'metadata' => [ ...arbitrary training config... ]
 *     ]
 *
 * Works identically for CPU and GPU models — `Parameter::data()`
 * always returns the PHP-side float array, regardless of which
 * backend is currently active.
 */
final class ModelSerializer
{
    public function __construct(
        private Serializer $serializer = new JsonSerializer(),
    ) {}

    /**
     * Convenience factory — picks a serializer based on file extension.
     *
     *   ModelSerializer::forFile('mnist.safetensors')  → SafeTensorsSerializer
     *   ModelSerializer::forFile('mnist.json')         → JsonSerializer
     */
    public static function forFile(string $path): self
    {
        if (str_ends_with($path, '.safetensors')) {
            return new self(new SafeTensorsSerializer());
        }
        return new self(new JsonSerializer());
    }

    /**
     * Save a model's parameters to disk.
     *
     * @param Module $model     Any module with parameters (Sequential, TransformerLM, ...)
     * @param string $path      Destination file
     * @param array  $metadata  Optional training config, hyperparameters, etc.
     */
    public function save(Module $model, string $path, array $metadata = []): void
    {
        $state = ['layers' => []];

        foreach ($model->parameters() as $i => $p) {
            $state['layers']["param_{$i}"] = [
                'shape' => $p->shape()->dims(),
                'data'  => $p->data(),
            ];
        }

        if (!empty($metadata)) {
            $state['metadata'] = $metadata;
        }

        $this->serializer->save($state, $path);
    }

    /**
     * Load parameter values into the model's existing parameters.
     *
     * The model's architecture (layer count, shapes) must already match
     * the saved checkpoint. Shape mismatches raise a RuntimeException.
     *
     * @param Module $model  Model to load into (in place)
     * @param string $path   Source file
     *
     * @return array  The metadata stored with the checkpoint (may be empty).
     */
    public function load(Module $model, string $path): array
    {
        $state = $this->serializer->load($path);

        if (!isset($state['layers']) || !is_array($state['layers'])) {
            throw new \RuntimeException(
                "Invalid checkpoint: missing 'layers' section."
            );
        }

        $params = $model->parameters();

        // Reach into Tensor::$data via reflection to write in place.
        // The Parameter subclass inherits this property from Tensor.
        $ref = new \ReflectionClass(Tensor::class);
        $dataProp = $ref->getProperty('data');
        $dataProp->setAccessible(true);

        foreach ($params as $i => $p) {
            $key = "param_{$i}";

            if (!isset($state['layers'][$key])) {
                throw new \RuntimeException(
                    "Missing parameter '{$key}' in checkpoint."
                );
            }

            $saved = $state['layers'][$key];

            if (!isset($saved['shape'], $saved['data'])) {
                throw new \RuntimeException(
                    "Malformed entry '{$key}' in checkpoint (missing shape or data)."
                );
            }

            if ($saved['shape'] !== $p->shape()->dims()) {
                throw new \RuntimeException(
                    "Shape mismatch for {$key}: saved [" .
                    implode(',', $saved['shape']) .
                    "] vs current [" .
                    implode(',', $p->shape()->dims()) . "]"
                );
            }

            if (count($saved['data']) !== $p->shape()->size()) {
                throw new \RuntimeException(
                    "Data size mismatch for {$key}: " .
                    count($saved['data']) . " values vs expected " .
                    $p->shape()->size() . "."
                );
            }

            $dataProp->setValue(
                $p,
                array_map('floatval', $saved['data'])
            );
        }

        return $state['metadata'] ?? [];
    }
}