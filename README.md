# Character Prefix Conditioning with Back Tokenization

This repository implements a solution for [Character Prefix Conditioning](https://www.cursor.com/blog/cpc) when working with language model tokenizers.

Read about the implementation in my [blog post](https://anilturaga.github.io/cpc).

## Overview

High-Level Approach
The strategy involves four main steps:

1. Create a trie data structure from the tokenizer vocabulary with the ability to take a string and get all tokens that has the the string as the prefix
2. For a given user input, search from the right to left of the last work determined by the regex pattern. Move character by character from left to right accumulating the prefix and for each prefix, find possible token completions that have the prefix
3. Filter token candidates to ensure they maintain valid token boundaries
4. Return a list of possible completion paths for the model to consider

### Trie Implementation
At the core of our solution is a trie (prefix tree) data structure optimized for byte-level searching.
Each node in the trie represents a byte and the children of the node represent the next byte in the token.
The trie also stores the original token that ends at that node.

```
Visualizing subtree for prefix 'user':

==================================================
TRIE VISUALIZATION (max_depth=4, max_children=3)
==================================================
Visualizing subtree for prefix: b'user'
* TOKENS: user
├── 'D'
│   └── 'a'
│       └── 't'
│           └── 'a'
│               * TOKENS: userData
├── 'I'
│   ├── 'D'
│   │   * TOKENS: userID
│   ├── 'd'
│   │   * TOKENS: userId
│   └── 'n'
│       └── 'f'
│           └── 'o'
│               * TOKENS: userInfo
└── 'M'
    └── 'a'
        └── 'n'
            └── 'a'
                └── 'g'
                    └── ... (more nodes, reached max depth)
└── ... (8 more children)
==================================================
```
So given a prefix, we can traverse the trie and collect all the tokens that have the prefix.

### Token Matching Algorithm

We first take the user's input and split it into words using the [regex pattern from the tokenizer library](https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken_ext/openai_public.py#L89).

For example, for the user sentence: `He introduced an intermediar`, the last words would be ` intermediar`.

But for explaining the process, we'll use the simpler user sentence: `indivi`.

For the user sentence: "indivi", we search the trie for each of these prefixes. We would initiate search for the following prefixes:
```
i
vi
ivi
divi
ndivi
indivi
_indivi
```

### Token Boundary Validation
The above search yields many potential candidates, forming a superset of the combinations we want to explore. We need an effective pruning strategy to filter these combinations.

Token Validation Strategy
For each candidate token, we:
* Replace the last word's prefix with the candidate token
* Re-encode the resulting string
* Compare with expected token sequence
* Keep only if the sequences match exactly


Let's walk through an example using the prefix "divi" from our search candidates above:

Example: Validating token "division"
Input Analysis:
* Last word: " indivi"
* Prefix: "divi"
* Candidate token: "division"
* Resulting word: " indivision"

The re-encoded tokens don't match our expected sequence, we discard "division" as a candidate. By applying this validation process across all candidate tokens, we can effectively filter out invalid completions while preserving the ones that maintain proper token boundaries.

## Examples

Consider these example inputs and how the algorithm processes them:

```
Testing: I bought some apple
last word  apple
Found 1424 matching tokens in 1.25 ms
Analysis completed in 19.13 ms
Combinations:
Pos 19: I bought some appl -> [b'en', b'ers', b'ere']... 474 possible tokens
Pos 18: I bought some app -> [b'led', b'lem', b'let']... 101 possible tokens
Pos 17: I bought some ap -> [b'pler']... 1 possible tokens
Pos 14: I bought some -> [b' apple', b' apples']... 2 possible tokens

Testing: https:
last word :
Found 324 matching tokens in 0.36 ms
Analysis completed in 2.42 ms
Combinations:
Pos 6: https -> [b':', b'::', b':\n']... 324 possible tokens

Testing: userNa
last word userNa
Found 2323 matching tokens in 1.90 ms
Analysis completed in 21.15 ms
Combinations:
Pos 6: userN -> [b'an', b'ar', b'al']... 2170 possible tokens
Pos 5: user -> [b'Na', b'Nam', b'Nav']... 34 possible tokens
Pos 1:  -> [b'userName']... 1 possible tokens

Testing: We found a hidden causali
last word  causali
Found 2106 matching tokens in 1.67 ms
Analysis completed in 21.13 ms
Combinations:
Pos 25: We found a hidden causal -> [b'it', b'io', b'id']... 1748 possible tokens
Pos 23: We found a hidden caus -> [b'ali', b'alin', b'alia']... 17 possible tokens

Testing: He introduced an intermediar
last word  intermediar
Found 922 matching tokens in 0.81 ms
Analysis completed in 11.40 ms
Combinations:
Pos 28: He introduced an intermedia -> [b'res', b'ress', b'resp']... 52 possible tokens
Pos 27: He introduced an intermedi -> [b'ar', b'art', b'are']... 215 possible tokens
Pos 17: He introduced an -> [b' intermediary']... 1 possible tokens

Testing: indivi
last word indivi
Found 2106 matching tokens in 1.37 ms
Analysis completed in 19.37 ms
Combinations:
Pos 6: indiv -> [b'in', b'it', b'is']... 1885 possible tokens
Pos 4: ind -> [b'ivi', b'ivil', b'ivid']... 16 possible tokens
Pos 1:  -> [b'individual']... 1 possible tokens
```


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
