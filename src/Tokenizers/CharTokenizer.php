<?php

declare(strict_types=1);

namespace ZillaPHP\Tokenizers;

/**
 * Simplest possible tokenizer: each character is one token.
 *
 *   'a' → 3, 'b' → 4, ...
 *
 * For a first language model this is exactly what we want — no
 * subword algorithm, no vocabulary training, just a bijection
 * between characters and integers.
 */
final class CharTokenizer
{
    /** @var array<string,int> */
    private array $charToId;

    /** @var array<int,string> */
    private array $idToChar;

    public function __construct(string $text)
    {
        $chars = array_values(array_unique(str_split($text)));
        sort($chars);              // deterministic ordering

        $this->charToId = [];
        $this->idToChar = [];

        foreach ($chars as $i => $c) {
            $this->charToId[$c] = $i;
            $this->idToChar[$i] = $c;
        }
    }

    public function vocabSize(): int
    {
        return count($this->charToId);
    }

    /** @return int[] */
    public function encode(string $text): array
    {
        $out = [];
        $len = strlen($text);
        for ($i = 0; $i < $len; $i++) {
            $c = $text[$i];
            if (!isset($this->charToId[$c])) {
                throw new \RuntimeException("Unknown character: " . bin2hex($c));
            }
            $out[] = $this->charToId[$c];
        }
        return $out;
    }

    /** @param int[] $ids */
    public function decode(array $ids): string
    {
        $out = '';
        foreach ($ids as $id) {
            if (!isset($this->idToChar[$id])) {
                throw new \RuntimeException("Unknown token id: {$id}");
            }
            $out .= $this->idToChar[$id];
        }
        return $out;
    }

    /** @return array{0:string[],1:string[]} */
    public function vocabulary(): array
    {
        return [array_keys($this->charToId), array_values($this->idToChar)];
    }
}