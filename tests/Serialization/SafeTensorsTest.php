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

    public function test_round_trip_preserves_f32_weights(): void
    {
        $model = new Sequential([
            new Linear(8, 4),
            new ReLU(),
            new Linear(4, 2),
        ]);

        $tmp = sys_get_temp_dir() . '/zilla_test_' . uniqid() . '.safetensors';
        $serializer = new ModelSerializer(new SafeTensorsSerializer());
        $serializer->save($model, $tmp, ['test' => 'roundtrip']);

        $this->assertFileExists($tmp);

        $model2 = new Sequential([
            new Linear(8, 4),
            new ReLU(),
            new Linear(4, 2),
        ]);

        $meta = $serializer->load($model2, $tmp);
        $this->assertSame(['test' => 'roundtrip'], $meta);

        // Compare in float32 space (bit-exact)
        $origParams   = $model->parameters();
        $loadedParams = $model2->parameters();

        foreach ($origParams as $i => $p) {
            $orig = $p->data();
            $back = $loadedParams[$i]->data();

            $this->assertCount(count($orig), $back);

            foreach ($orig as $j => $v) {
                $expected = unpack('f', pack('f', $v))[1];
                $this->assertEqualsWithDelta(
                    $expected,
                    $back[$j],
                    1e-9,
                    "Parameter {$i} index {$j} differs beyond float32 precision."
                );
            }
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