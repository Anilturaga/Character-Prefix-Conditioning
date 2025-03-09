from typing import Dict, List, Optional
import re
import tiktoken
from tiktoken._educational import *
enc = tiktoken.get_encoding("cl100k_base")
len(enc._mergeable_ranks.keys())
list(enc._mergeable_ranks.keys())[-10]
enc._mergeable_ranks
vocab = enc._mergeable_ranks
# tokenizer = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
# vocab = tokenizer.mergeable_ranks


def find_matching_tokens(prefix: str, vocab: Dict[bytes, int]) -> List[bytes]:
    """
    Find all tokens in vocabulary that start with the given prefix
    
    Args:
        prefix: String prefix to match against
        vocab: Dictionary mapping token bytes to token ids
        
    Returns:
        List of tokens that match the prefix
    """
    # Convert prefix to bytes for matching
    prefix_bytes = prefix.encode('utf-8')
    
    # Find all matching tokens
    matches = []
    for token in vocab.keys():
        # Convert token to string representation
        if token.startswith(b'\xc4\xa0'): # Ġ in UTF-8
            token_str = b' ' + token[2:] 
        elif token.startswith(b'##'):
            token_str = token[2:]
        else:
            token_str = token
            
        if token_str.startswith(prefix_bytes):
            matches.append(token)
            
    return matches

def find_tokens_from_right(sentence: str, vocab: Dict[bytes, int]) -> List[List[bytes]]:
    """
    Find matching tokens by taking progressively longer prefixes from the right side
    
    Args:
        sentence: Input sentence to analyze
        vocab: Dictionary mapping token bytes to token ids
        
    Returns:
        List of lists, where each inner list contains tokens matching the prefix
        starting from that position
    """
    matches_by_position = []
    
    # Start from the end and work backwards
    for i in range(len(sentence)):
        prefix = sentence[-(i+1):]  # Take i+1 chars from the right
        matching_tokens = find_matching_tokens(prefix, vocab)
        
        # Filter tokens based on whether they maintain token boundaries when re-encoded
        filtered_tokens = []
        for token in matching_tokens:
            # Convert token to string representation
            if token.startswith(b'\xc4\xa0'): # Ġ in UTF-8
                token_str = b' ' + token[2:]
            elif token.startswith(b'##'):
                token_str = token[2:]
            else:
                token_str = token
                
            # Create test sentence by replacing prefix with this token
            test_str = sentence[:-(i+1)].encode('utf-8') + token_str
            # print(token_str,":",test_str)
            # Re-encode the test sentence
            test_tokens = enc.encode(test_str.decode('utf-8', errors='ignore'))
            
            # Get tokens for original sentence up to the replacement point
            original_tokens = enc.encode(sentence[:-(i+1)])
            # Combine with our test token
            expected_tokens = original_tokens + test_tokens[-1:]
            # Re-encode full test string to compare
            actual_tokens = enc.encode(test_str.decode('utf-8', errors='ignore'))
            # print(expected_tokens,":",actual_tokens)
            # Check if tokens match except for the last one we replaced
            if expected_tokens == actual_tokens:
                # print("Match")
                filtered_tokens.append(token)
        matching_tokens = filtered_tokens
        matches_by_position.append(matching_tokens)
        
    return matches_by_position

def analyze_token_combinations(sentence: str, vocab: Dict[bytes, int]) -> List[Dict]:
    """
    Analyze possible token combinations for a given sentence
    
    Args:
        sentence: Input sentence to analyze
        vocab: Dictionary mapping token bytes to token ids
        
    Returns:
        List of dictionaries containing position, prefix and matching tokens
    """
    right_matches = find_tokens_from_right(sentence, vocab)
    combinations = []
    
    for i, matches in enumerate(right_matches):
        if len(matches) > 0:
            combinations.append({
                'position': len(sentence) - i,
                'prefix': sentence[:-(i+1)],
                'matches': matches
            })
            
    return combinations

if __name__ == "__main__":
    # Example usage
    test_cases = [
        "The agreement was signed unconditiona",
        "He introduced an intermediar",
        "We found a hidden correla", 
        "I bought some apple",
        "I am an indivi",
        "I am indivi"
    ]

    for test_sentence in test_cases:
        print(f"\nTesting: {test_sentence}")
        combinations = analyze_token_combinations(test_sentence, vocab)
                
        print("Combinations:")
        for c in combinations:
            print(f"Pos {c['position']}: {c['prefix']} -> {c['matches'][:3]}... {len(c['matches'])} possible tokens")
