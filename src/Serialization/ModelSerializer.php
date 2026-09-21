<?php

declare(strict_types=1);

namespace ZillaPHP\Serialization;

use ZillaPHP\NN\Module;
use ZillaPHP\NN\Parameter;
use ZillaPHP\Tensor\Shape;

final class ModelSerializer
{
    public function __construct(private Serializer $serializer = new JsonSerializer()) {}

    public function save(Module $model, string $path, array $metadata = []): void
    {
        $state = ['layers' => []];

        foreach ($model->parameters() as $i => $p) {
            $state['layers']["param_{$i}"] = [
                'shape' => $p->shape()->dims(),
                'data'  => $p->data(),
            ];
        }

        $state['metadata'] = $metadata;

        $this->serializer->save($state, $path);
    }

    /**
     * Loads parameter values from disk into the model's existing parameters.
     * The model architecture must already match the saved checkpoint.
     */
    public function load(Module $model, string $path): void
    {
        $state = $this->serializer->load($path);

        if (!isset($state['layers']) || !is_array($state['layers'])) {
            throw new \RuntimeException("Invalid checkpoint: missing 'layers'.");
        }

        $params = $model->parameters();
        $ref = new \ReflectionClass(\ZillaPHP\Tensor\Tensor::class);
        $dataProp = $ref->getProperty('data');
        $dataProp->setAccessible(true);

        foreach ($params as $i => $p) {
            $key = "param_{$i}";
            if (!isset($state['layers'][$key])) {
                throw new \RuntimeException("Missing parameter {$key} in checkpoint.");
            }
            $saved = $state['layers'][$key];

            if ($saved['shape'] !== $p->shape()->dims()) {
                throw new \RuntimeException(
                    "Shape mismatch for {$key}: saved [" . implode(',', $saved['shape']) .
                    "] vs current [" . implode(',', $p->shape()->dims()) . "]"
                );
            }

            $dataProp->setValue($p, array_map('floatval', $saved['data']));
        }
    }
}