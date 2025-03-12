# Character Prefix Conditioning with Back Tokenization
Attempt to solve [Cursor's challenge on code completion sampling](https://www.cursor.com/blog/cpc).

Read `prompt.txt` for better understanding of the problem.

This project implements an efficient algorithm for finding token completions possibilities from a incomplete sentence.
We essentially take the last word of the sentence and backtrack over each byte from the back.
For each byte, we find the tokens that have the byte and anything right of it as a prefix.
We then check if the token is valid by encoding the sentence before the byte and the token and checking if the encoding is the same as the encoding of the sentence. This will drastically reduce the number of completion combinations we have to generate for.

It's designed to improve the performance of prefix matching for completion models.

## Overview

The implementation consists of two main components:

1. **Token Matching Algorithm**: Finds tokens in a vocabulary that match a given prefix
2. **Regex for word boundary detection**: Used to split the sentence into words
3. **Byte Trie**: A trie data structure optimized for byte-level prefix matching


## Files

- `main.py`: Contains the full implementation of both the token analysis algorithm and the optimized trie data structure
- `benchmark.py`: Tools to measure performance improvements between the original approach and the trie-based solution

## Usage

To run the token analysis with the trie-based implementation:

```python
import tiktoken
from main import analyze_token_combinations, init_trie

# Initialize the tokenizer and vocabulary
enc = tiktoken.get_encoding("cl100k_base")
vocab = enc._mergeable_ranks

# Initialize the trie (done once)
trie = init_trie(vocab)

# Analyze token combinations for a sentence
sentence = "The agreement was signed unconditiona"
combinations = analyze_token_combinations(sentence, vocab, trie)

# Print results
for c in combinations:
    print(f"Position {c['position']}: {c['prefix']} -> {len(c['matches'])} possible tokens")
```

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
