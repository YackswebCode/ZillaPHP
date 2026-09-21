<?php

declare(strict_types=1);

namespace ZillaPHP\CLI;

use ZillaPHP\Generators\Sampler;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Tokenizers\CharTokenizer;
use ZillaPHP\Transformers\TransformerLM;

final class Application
{
    public const VERSION = '0.1.0-dev';

    public function run(array $argv): int
    {
        $command = $argv[1] ?? 'help';

        return match ($command) {
            'doctor'        => $this->doctor(),
            'version'       => $this->version(),
            'list'          => $this->listCommands(),
            'help'          => $this->help(),
            'device:list'   => $this->deviceList(),
            'train'         => $this->train($argv),
            'generate:text' => $this->generateText($argv),
            default         => $this->unknown($command),
        };
    }

    // ==================================================================
    // Diagnostic commands
    // ==================================================================

    private function doctor(): int
    {
        echo "ZillaPHP Doctor\n";
        echo str_repeat('-', 44) . "\n";
        printf("PHP version    : %s\n", PHP_VERSION);
        printf("ZillaPHP       : %s\n", self::VERSION);
        printf("OS             : %s\n", PHP_OS_FAMILY);
        printf("Architecture   : %s\n", php_uname('m'));
        printf("CPU cores      : %d\n", $this->cpuCores());
        printf("Memory limit   : %s\n", ini_get('memory_limit'));
        printf("FFI enabled    : %s\n", extension_loaded('FFI') ? 'yes' : 'no');
        printf("CPU lib        : %s\n",
            is_file(__DIR__ . '/../../native/cpu/libzilla_cpu.so') ? 'present' : 'not built');
        printf("CUDA lib       : %s\n",
            is_file(__DIR__ . '/../../native/cuda/libzilla_cuda.so') ? 'present' : 'not built');

        $cudaAvailable = false;
        if (is_file(__DIR__ . '/../../native/cuda/libzilla_cuda.so')) {
            try {
                $cuda = new \ZillaPHP\Hardware\CUDA\CudaBackend();
                $cudaAvailable = $cuda->isCuda();
            } catch (\Throwable) {}
        }
        printf("CUDA available : %s\n", $cudaAvailable ? 'yes' : 'no');
        echo "\nRecommended backend: CPU\n";
        return 0;
    }

    private function deviceList(): int
    {
        echo "Devices:\n";
        echo "  CPU  available\n";
        echo "  CUDA not available\n";
        echo "  ROCm not available\n";
        return 0;
    }

    private function cpuCores(): int
    {
        if (is_file('/proc/cpuinfo')) {
            return substr_count(file_get_contents('/proc/cpuinfo'), 'processor');
        }
        return 1;
    }

    private function version(): int
    {
        echo "ZillaPHP v" . self::VERSION . "\n";
        return 0;
    }

    // ==================================================================
    // Training
    // ==================================================================

    private function train(array $argv): int
    {
        $file = $argv[2] ?? null;
        if ($file === null) {
            fwrite(STDERR, "Usage: zilla train <example.php>\n");
            return 1;
        }

        $path = realpath($file);
        if ($path === false || !is_file($path)) {
            fwrite(STDERR, "File not found: {$file}\n");
            return 1;
        }

        echo "zilla train — running {$path}\n";
        echo str_repeat('=', 60) . "\n";

        $start = microtime(true);
        require $path;
        $elapsed = microtime(true) - $start;

        echo str_repeat('=', 60) . "\n";
        printf("Done in %.2f s\n", $elapsed);
        return 0;
    }

    // ==================================================================
    // Text generation
    // ==================================================================

    private function generateText(array $argv): int
    {
        $opts = $this->parseOptions(array_slice($argv, 2));

        $modelPath   = $opts['model']  ?? null;
        $prompt      = $opts['prompt'] ?? 'ROMEO: ';
        $maxTokens   = (int)   ($opts['tokens'] ?? 200);
        $temperature = (float) ($opts['temp']   ?? 0.8);
        $seed        = (int)   ($opts['seed']   ?? 42);

        if ($modelPath === null) {
            fwrite(STDERR,
                "Usage: zilla generate:text --model=<checkpoint.json> " .
                "[--prompt=\"...\"] [--tokens=200] [--temp=0.8] [--seed=42]\n"
            );
            return 1;
        }

        $checkpointPath = realpath($modelPath);
        if ($checkpointPath === false || !is_file($checkpointPath)) {
            fwrite(STDERR, "Checkpoint not found: {$modelPath}\n");
            return 1;
        }

        // ---- Read checkpoint metadata ----
        $raw  = file_get_contents($checkpointPath);
        $data = json_decode($raw, true);
        if (!is_array($data) || ($data['format'] ?? null) !== 'zilla-json') {
            fwrite(STDERR, "Not a valid ZillaPHP checkpoint.\n");
            return 1;
        }
        $meta = $data['state']['metadata'] ?? [];

        $vocabSize  = (int) ($meta['vocab_size']  ?? 65);
        $seqLen     = (int) ($meta['seq_len']     ?? 64);
        $dim        = (int) ($meta['dim']         ?? 64);
        $heads      = (int) ($meta['heads']       ?? 4);
        $layers     = (int) ($meta['layers']      ?? 2);
        $trainChars = (int) ($meta['train_chars'] ?? 50000);

        // ---- Rebuild the tokenizer from the original text ----
        $textFile = __DIR__ . '/../../data/shakespeare.txt';
        if (!is_file($textFile)) {
            fwrite(STDERR, "Cannot find data/shakespeare.txt to rebuild tokenizer.\n");
            fwrite(STDERR, "Run: php scripts/download_shakespeare.php\n");
            return 1;
        }
        $text = file_get_contents($textFile);
        $text = substr($text, 0, $trainChars);
        $tokenizer = new CharTokenizer($text);

        // ---- Rebuild the model architecture ----
        $model = new TransformerLM(
            vocabSize:  $vocabSize,
            dim:        $dim,
            heads:      $heads,
            layers:     $layers,
            maxSeqLen:  $seqLen,
            ffnHidden:  $dim * 2,
        );

        // ---- Load weights ----
        try {
            (new ModelSerializer())->load($model, $checkpointPath);
        } catch (\Throwable $e) {
            fwrite(STDERR, "Failed to load checkpoint: " . $e->getMessage() . "\n");
            return 1;
        }

        // ---- Generate ----
        $sampler = new Sampler($model, $tokenizer, $seqLen);

        echo "ZillaPHP — Text generation\n";
        echo str_repeat('=', 60) . "\n";
        echo "Model:       " . basename($checkpointPath) . "\n";
        echo "Vocab size:  {$vocabSize}\n";
        echo "Context:     {$seqLen}\n";
        echo "Prompt:      " . json_encode($prompt) . "\n";
        echo "Tokens:      {$maxTokens}\n";
        echo "Temperature: {$temperature}\n";
        echo str_repeat('-', 60) . "\n\n";

        $output = $sampler->generate(
            $prompt,
            maxNewTokens: $maxTokens,
            temperature:  $temperature,
            seed:         $seed,
        );

        echo $output . "\n\n";
        return 0;
    }

    /** @return array<string,string> */
    private function parseOptions(array $args): array
    {
        $out = [];
        foreach ($args as $arg) {
            if (preg_match('/^--([^=]+)=(.*)$/s', $arg, $m)) {
                $out[$m[1]] = $m[2];
            }
        }
        return $out;
    }

    // ==================================================================
    // Help / listing
    // ==================================================================

    private function listCommands(): int
    {
        echo "Available commands:\n";
        foreach ([
            'doctor',
            'device:list',
            'train <file>',
            'generate:text --model=<path> [--prompt="..."] [--tokens=200] [--temp=0.8]',
            'version',
            'list',
            'help',
        ] as $c) {
            echo "  zilla {$c}\n";
        }
        return 0;
    }

    private function help(): int
    {
        echo "ZillaPHP CLI\n\n";
        echo "Usage:\n  zilla <command> [options]\n\n";
        return $this->listCommands();
    }

    private function unknown(string $cmd): int
    {
        fwrite(STDERR, "Unknown command: {$cmd}\n");
        return 1;
    }
}