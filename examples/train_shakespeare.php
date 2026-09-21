<?php

require __DIR__ . '/../vendor/autoload.php';

use ZillaPHP\Core\Application;
use ZillaPHP\Generators\Sampler;
use ZillaPHP\Loss\CrossEntropy;
use ZillaPHP\Optim\Adam;
use ZillaPHP\Serialization\ModelSerializer;
use ZillaPHP\Tokenizers\CharTokenizer;
use ZillaPHP\Tensor\Tensor;
use ZillaPHP\Transformers\TransformerLM;

Application::boot(native: true);

// ====================================================================
// Data configuration
// ====================================================================
$TEXT_FILE       = __DIR__ . '/../data/shakespeare.txt';
$TRAIN_CHARS     = (int)   (getenv('TRAIN_CHARS')     ?: 50000);
$SEQ_LEN         = (int)   (getenv('SEQ_LEN')         ?: 32);
$EPOCHS          = (int)   (getenv('EPOCHS')          ?: 1);
$LR              = (float) (getenv('LR')              ?: 0.003);
$STEPS_PER_EPOCH = (int)   (getenv('STEPS_PER_EPOCH') ?: 100);
$LOG_EVERY       = (int)   (getenv('LOG_EVERY')       ?: 20);
$BATCH           = (int)   (getenv('BATCH')           ?: 1);

// ====================================================================
// Model configuration
// ====================================================================
$DIM    = (int) (getenv('DIM')    ?: 64);
$HEADS  = (int) (getenv('HEADS')  ?: 4);
$LAYERS = (int) (getenv('LAYERS') ?: 2);
$FFN    = (int) (getenv('FFN')    ?: $DIM * 2);

$CHECKPOINT = __DIR__ . '/../checkpoints/shakespeare.zilla.json';

if (!is_file($TEXT_FILE)) {
    fwrite(STDERR, "Shakespeare file not found at {$TEXT_FILE}.\n");
    fwrite(STDERR, "Run: php scripts/download_shakespeare.php\n");
    exit(1);
}

$text = file_get_contents($TEXT_FILE);
$text = substr($text, 0, $TRAIN_CHARS);

echo "ZillaPHP — Tiny Shakespeare LM\n";
echo str_repeat('=', 66) . "\n";
printf("Training chars: %d   Sequence length: %d\n", $TRAIN_CHARS, $SEQ_LEN);
printf("Epochs: %d   Steps/epoch: %d   LR: %.4f\n", $EPOCHS, $STEPS_PER_EPOCH, $LR);
printf("Batch size: %d\n", $BATCH);
printf("Model: dim=%d  heads=%d  layers=%d  ffn=%d\n", $DIM, $HEADS, $LAYERS, $FFN);
printf("Backend: %s\n", Tensor::backend()->name());
echo str_repeat('=', 66) . "\n\n";

// ---- Tokenizer ----------------------------------------------------
$tokenizer = new CharTokenizer($text);
$ids = $tokenizer->encode($text);
echo "Vocab size: {$tokenizer->vocabSize()}\n";
echo "Total tokens: " . count($ids) . "\n\n";

// ---- Model --------------------------------------------------------
$model = new TransformerLM(
    vocabSize:  $tokenizer->vocabSize(),
    dim:        $DIM,
    heads:      $HEADS,
    layers:     $LAYERS,
    maxSeqLen:  $SEQ_LEN,
    ffnHidden:  $FFN,
);

$params = $model->parameters();
$totalParams = array_sum(array_map(fn($p) => $p->shape()->size(), $params));
echo "Params: " . count($params) . " tensors, {$totalParams} numbers\n\n";

$optimizer  = new Adam($params, lr: $LR);
$loss       = new CrossEntropy();
$serializer = new ModelSerializer();

// ---- Ensure checkpoint dir exists ---------------------------------
$ckptDir = dirname($CHECKPOINT);
if (!is_dir($ckptDir)) mkdir($ckptDir, 0777, true);

// ---- Training loop ------------------------------------------------
$startTime   = microtime(true);
$globalStep  = 0;
$runningLoss = 0.0;
$n = count($ids);

for ($epoch = 0; $epoch < $EPOCHS; $epoch++) {
    for ($step = 0; $step < $STEPS_PER_EPOCH; $step++) {

        if ($BATCH === 1) {
            // ---- Single-sample path (identical to previous behaviour) ----
            $start = mt_rand(0, $n - $SEQ_LEN - 1);
            $inputIds  = array_slice($ids, $start, $SEQ_LEN);
            $targetIds = array_slice($ids, $start + 1, $SEQ_LEN);

            $input = Tensor::fromArray(array_map('floatval', $inputIds));

            $optimizer->zeroGrad();
            $logits = $model->forward($input);
            $l      = $loss->forward($logits, $targetIds);
        } else {
            // ---- Batched path ----
            $inputBatch  = [];
            $flatTargets = [];
            for ($b = 0; $b < $BATCH; $b++) {
                $start = mt_rand(0, $n - $SEQ_LEN - 1);
                $row   = array_slice($ids, $start, $SEQ_LEN);
                $inputBatch[] = array_map('floatval', $row);

                $tgtRow = array_slice($ids, $start + 1, $SEQ_LEN);
                foreach ($tgtRow as $t) $flatTargets[] = $t;
            }

            $input = Tensor::fromArray($inputBatch);   // [B, T]

            $optimizer->zeroGrad();
            $logits = $model->forwardBatch($input);    // [B*T, vocab]
            $l      = $loss->forward($logits, $flatTargets);
        }

        $l->backward();
        $optimizer->step();

        $runningLoss += (float) $l->item();
        $globalStep++;

        if ($globalStep % $LOG_EVERY === 0) {
            $avg     = $runningLoss / $LOG_EVERY;
            $elapsed = microtime(true) - $startTime;
            printf("epoch %d  step %4d  loss=%.4f  %.1fs\n",
                $epoch, $step, $avg, $elapsed);
            $runningLoss = 0.0;
        }
    }

    // ---- Sample after each epoch ----------------------------------
    echo "\n----- Sample after epoch {$epoch} -----\n";
    $model->eval();
    $sampler = new Sampler($model, $tokenizer, $SEQ_LEN);
    echo $sampler->generate("ROMEO: ", maxNewTokens: 200, temperature: 0.8, seed: 42) . "\n";
    $model->train();

    // ---- Save checkpoint ------------------------------------------
    $serializer->save($model, $CHECKPOINT, [
        'epoch'       => $epoch,
        'vocab_size'  => $tokenizer->vocabSize(),
        'seq_len'     => $SEQ_LEN,
        'dim'         => $DIM,
        'heads'       => $HEADS,
        'layers'      => $LAYERS,
        'train_chars' => $TRAIN_CHARS,
    ]);
    echo "\nSaved: {$CHECKPOINT}\n\n";
}

$total = microtime(true) - $startTime;
printf("Total: %.1f s (%.1f min)\n", $total, $total / 60);