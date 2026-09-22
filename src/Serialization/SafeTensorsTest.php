<?php

declare(strict_types=1);

namespace ZillaPHP\Tests\Serialization;

use PHPUnit\Framework\TestCase;
use ZillaPHP\Core\Application;
use ZillaPHP\NN\Activations\ReLU;
use ZillaPHP\NN\Layers\Linear;
use ZillaPHP\NN\Sequential;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Serialization\SafeTensorsSerializer;

final class SafeTensorsTest extends TestCase
{
    protected function setUp(): void
    {
        Application::boot(native: true);
    }

    public function test_round_trip_preserves_weights(): void
    {
        // Build a small model
        $model = new Sequential([
            new Linear(8, 4),
            new ReLU(),
            new Linear(4, 2),
        ]);

        // Snapshot original weights
        $original = array_map(
            fn($p) => $p->data(),
            $model->parameters()
        );

        // Save
        $tmp = sys_get_temp_dir() . '/zilla_test_' . uniqid() . '.safetensors';
        $serializer = new ModelSerializer(new SafeTensorsSerializer());
        $serializer->save($model, $tmp, ['test' => 'roundtrip']);

        $this->assertFileExists($tmp);

        // Rebuild model with fresh random weights
        $model2 = new Sequential([
            new Linear(8, 4),
            new ReLU(),
            new Linear(4, 2),
        ]);

        // Load
        $serializer->load($model2, $tmp);

        // Verify bit-exact
        $loaded = array_map(
            fn($p) => $p->data(),
            $model2->parameters()
        );

        $this->assertCount(count($original), $loaded);

        foreach ($original as $i => $origData) {
            $this->assertSame(
                $origData,
                $loaded[$i],
                "Parameter {$i} was not restored bit-exactly."
            );
        }

        unlink($tmp);
    }

    public function test_header_format_is_valid(): void
    {
        $model = new Sequential([new Linear(3, 2)]);
        $tmp = sys_get_temp_dir() . '/zilla_hdr_' . uniqid() . '.safetensors';

        (new ModelSerializer(new SafeTensorsSerializer()))->save($model, $tmp);

        $raw = file_get_contents($tmp);
        $headerSize = unpack('P', substr($raw, 0, 8))[1];

        // Header must be 8-byte aligned
        $this->assertSame(0, $headerSize % 8, "Header size not 8-byte aligned.");

        $headerJson = substr($raw, 8, $headerSize);
        $header = json_decode($headerJson, true);
        $this->assertIsArray($header);

        foreach ($header as $name => $info) {
            if ($name === '__metadata__') continue;

            $this->assertArrayHasKey('dtype', $info);
            $this->assertArrayHasKey('shape', $info);
            $this->assertArrayHasKey('data_offsets', $info);
            $this->assertSame('F32', $info['dtype']);
            $this->assertCount(2, $info['data_offsets']);
        }

        unlink($tmp);
    }
}