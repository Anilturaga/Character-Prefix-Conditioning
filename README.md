# Character Prefix Conditioning with Back Tokenization

This repository implements a solution for [Character Prefix Conditioning](https://www.cursor.com/blog/cpc) when working with language model tokenizers.

## Overview

When generating completions with language models, we need to ensure that the model produces text that begins with what the user has already typed. However, tokenizers operate on token boundaries, not character boundaries, creating challenges when the user's cursor doesn't lie on a token boundary.

This implementation provides an efficient algorithm for sampling tokens conditioned on a character prefix.

## Key Features

- **Trie-Based Token Matching**: Uses an optimized trie data structure for efficient prefix matching
- **Last Word Processing**: Focuses search on the last word of input for better performance. Last word is found using a regex pattern [implemented in the tiktoken library](https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken_ext/openai_public.py#L89)
- **Token Boundary Validation**: Ensures proper tokenization by validating token boundaries
- **Efficient Prefix Matching**: Works backwards from the end of the last word to find all possible token combinations. This might not be the best approach, but it works.

## How It Works

1. The implementation builds a trie data structure from the tokenizer vocabulary
2. When given an input string, it extracts the last word using a regex pattern
3. Starting from the right end of the last word, it progressively tries longer prefixes
4. For each prefix, it finds all matching tokens in the vocabulary
5. It validates each token by checking if it maintains proper token boundaries when re-encoded
6. Returns a list of valid tokens at each position that can be used for sampling

## Examples

Consider these example inputs and how the algorithm processes them:

### Example 1
Input: "The agreement was signed unconditiona"
- Last word: "unconditiona"
- Finds tokens that can validly complete this prefix (like "lly")
- Enables completion to "The agreement was signed unconditionally"

### Example 2
Input: "He introduced an intermediar"
- Last word: "intermediar"
- Finds "y" as a valid completion token
- Enables completion to "He introduced an intermediary"

### Example 3
Input: "I bought some apple"
- Last word: "apple"
- Finds "s" as a valid completion token
- Enables completion to "I bought some apples"

### Example 4
Input: "indivi"
- Last word: "indivi"
- Finds "dual" or "sible" as a valid completion token
- Enables completion to "individual" or "indivisible"

### Example 5
Input: "https:"
- Last word: "https:"
- Finds "://" as a valid completion token
- Enables completion to "https://www.google.com"

## Implementation Details

### Trie Structure

The implementation uses a specialized trie data structure to efficiently find all tokens that match a given prefix. The trie is built once from the tokenizer vocabulary and reused for all queries.

### Last Word Extraction

The implementation uses a regex pattern to split the input sentence into words and focuses only on the last word, which significantly improves performance by reducing the search space.

Pattern is specific to the cl100k_base tokenizer and can be found [here](https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken_ext/openai_public.py#L89)

### Token Boundary Validation

The implementation validates each potential token by checking if it maintains proper token boundaries when re-encoded:

1. Creates a test string by replacing the prefix with the candidate token
2. Re-encodes the test string
3. Compares the resulting tokens with the expected tokens
4. Only keeps tokens that maintain proper boundaries

This ensures that the suggested completions will work correctly when tokenized.

## Usage

Initialize the trie structure with your tokenizer vocabulary:

```python
trie = init_trie(vocab)
```

Analyze possible token completions for a sentence:

```python
combinations = analyze_token_combinations(sentence, vocab, trie)
```

The result contains positions, prefixes, and matching tokens that can be used for sampling.

## Files

- `main.py`: Contains the full implementation of both the token analysis algorithm and the optimized trie data structure
- `benchmark.py`: Tools to measure performance improvements between the original approach and the trie-based solution

## Benchmarking

Run the benchmark script to compare performance:

```
python benchmark.py
```

The benchmark measures:
- Simple prefix matching performance
- Full token combination analysis 
- Amortized cost including trie initialization

## Future Improvements

Potential future optimizations:
- replacing encoding and decoding steps for matching with something based on mergeable_ranks
- Implement a more memory-efficient trie structure
- Explore parallel processing for building the trie
- Add more aggressive caching strategies
- Optimize the token filtering process 
