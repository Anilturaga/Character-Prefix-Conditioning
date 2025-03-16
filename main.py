from typing import Dict, List, Optional
import re
import tiktoken
import time
from tiktoken._educational import *
from collections import deque

enc = tiktoken.get_encoding("cl100k_base")
len(enc._mergeable_ranks.keys())
list(enc._mergeable_ranks.keys())[-10]
enc._mergeable_ranks
vocab = enc._mergeable_ranks
# tokenizer = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
# vocab = tokenizer.mergeable_ranks

# Trie implementation for efficient prefix matching
class TrieNode:
    def __init__(self):
        self.children = {}  # Maps bytes to child nodes
        self.is_end_of_token = False
        self.tokens = []  # Original tokens that end at this node

class BytesTrie:
    """Optimized trie structure for efficient prefix matching"""
    
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, token_bytes: bytes, original_token: bytes):
        """
        Insert a token into the trie
        
        Args:
            token_bytes: The processed bytes to insert (e.g., after removing special prefixes)
            original_token: The original token as stored in the vocabulary
        """
        node = self.root
        for b in token_bytes:
            if b not in node.children:
                node.children[b] = TrieNode()
            node = node.children[b]
        
        node.is_end_of_token = True
        node.tokens.append(original_token)
    
    def _find_node(self, prefix_bytes: bytes) -> Optional[TrieNode]:
        """Find the node corresponding to a prefix"""
        node = self.root
        for b in prefix_bytes:
            if b not in node.children:
                return None
            node = node.children[b]
        return node
        
    def search_prefix(self, prefix_bytes: bytes) -> List[bytes]:
        """
        Find all tokens that start with the given prefix
        
        Args:
            prefix_bytes: Prefix to search for
            
        Returns:
            List of original tokens that match the prefix
        """
        node = self.root
        for b in prefix_bytes:
            if b not in node.children:
                return []  # Prefix not found
            node = node.children[b]
            
        # Found the prefix, now collect all tokens under this node
        return self._collect_tokens_iterative(node)
    
    def _collect_tokens_iterative(self, start_node: TrieNode) -> List[bytes]:
        """
        Collect all tokens under a given node iteratively (avoids stack overflow)
        """
        tokens = []
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            
            # Add tokens at current node
            if node.is_end_of_token:
                tokens.extend(node.tokens)
                
            # Add all children to the queue
            for child in node.children.values():
                queue.append(child)
                
        return tokens
        
    def print_trie(self, max_depth=3, max_children=5, max_tokens=3, start_prefix=b''):
        """
        Visualize the trie structure
        
        Args:
            max_depth: Maximum depth to visualize
            max_children: Maximum number of children to show at each node
            max_tokens: Maximum number of complete tokens to show at each node
            start_prefix: Optional starting prefix to visualize a subtree
        """
        print(f"\n{'='*50}")
        print(f"TRIE VISUALIZATION (max_depth={max_depth}, max_children={max_children})")
        print(f"{'='*50}")
        
        # Find the starting node
        if start_prefix:
            start_node = self._find_node(start_prefix)
            if not start_node:
                print(f"Prefix {start_prefix} not found in trie")
                return
            print(f"Visualizing subtree for prefix: {start_prefix}")
        else:
            start_node = self.root
            print("Visualizing from root")
            
        # Define recursive printing function
        def _print_node(node, prefix, depth, path=b''):
            if depth > max_depth:
                if node.children:
                    print(f"{prefix}└── ... (more nodes, reached max depth)")
                return
                
            # Print tokens at this node if it's end of token
            if node.is_end_of_token:
                token_str = ', '.join([t.decode('utf-8', errors='replace') for t in node.tokens[:max_tokens]])
                if len(node.tokens) > max_tokens:
                    token_str += f", ... ({len(node.tokens) - max_tokens} more)"
                print(f"{prefix}* TOKENS: {token_str}")
                
            # Print children
            items = list(node.children.items())
            if not items:
                return
                
            # Sort children by byte value for consistent output
            items.sort(key=lambda x: x[0])
            
            # Show limited number of children
            for i, (byte, child) in enumerate(items[:max_children]):
                is_last = i == len(items[:max_children]) - 1
                
                # Format the byte value
                try:
                    byte_str = f"'{chr(byte)}'" if 32 <= byte <= 126 else f"0x{byte:02x}"
                except:
                    byte_str = f"0x{byte:02x}"
                    
                # Print the node
                if is_last:
                    print(f"{prefix}└── {byte_str}")
                    new_prefix = prefix + "    "
                else:
                    print(f"{prefix}├── {byte_str}")
                    new_prefix = prefix + "│   "
                    
                # Recursively print children
                child_path = path + bytes([byte])
                _print_node(child, new_prefix, depth + 1, child_path)
                
            # Indicate if there are more children than shown
            if len(items) > max_children:
                print(f"{prefix}└── ... ({len(items) - max_children} more children)")
        
        # Start recursive printing
        _print_node(start_node, "", 0)
        print(f"{'='*50}")

# Global trie instance
vocab_trie = None

def build_trie_from_vocab(vocab: Dict[bytes, int]) -> BytesTrie:
    """
    Build a trie from a vocabulary
    
    Args:
        vocab: Dictionary mapping token bytes to token ids
        
    Returns:
        BytesTrie built from the vocabulary
    """
    print("Building trie from vocabulary...")
    start_time = time.time()
    
    trie = BytesTrie()
    for token in vocab.keys():
        # Process token for insertion
        if token.startswith(b'\xc4\xa0'):  # Ġ in UTF-8
            token_str = b' ' + token[2:]
        elif token.startswith(b'##'):
            token_str = token[2:]
        else:
            token_str = token
            
        trie.insert(token_str, token)
    
    build_time = time.time() - start_time
    print(f"Trie built in {build_time:.2f} seconds")
        
    total_time = time.time() - start_time
    print(f"Total initialization time: {total_time:.2f} seconds")
    
    return trie

def find_matching_tokens(prefix: str, vocab: Dict[bytes, int], trie=None) -> List[bytes]:
    """
    Find all tokens in vocabulary that start with the given prefix
    
    Args:
        prefix: String prefix to match against
        vocab: Dictionary mapping token bytes to token ids
        trie: Optional trie data structure for faster matching
        
    Returns:
        List of tokens that match the prefix
    """
    # Convert prefix to bytes for matching
    prefix_bytes = prefix.encode('utf-8')
    
    # Use trie if provided
    if trie is not None:
        return trie.search_prefix(prefix_bytes)
    
    # Fallback to original implementation
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

def find_tokens_from_right(sentence: str, vocab: Dict[bytes, int], trie=None) -> List[List[bytes]]:
    """
    Find matching tokens by taking progressively longer prefixes from the right side
    
    Args:
        sentence: Input sentence to analyze
        vocab: Dictionary mapping token bytes to token ids
        trie: Optional trie data structure for faster matching
        
    Returns:
        List of lists, where each inner list contains tokens matching the prefix
        starting from that position
    """
    matches_by_position = []
    
    # Split sentence into words using regex
    import regex
    pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
    words = regex.findall(pattern, sentence)
    
    # Get the last word
    if not words:
        return matches_by_position
        
    last_word = words[-1] # removed strip() to keep things like whitespace
    if not last_word:
        return matches_by_position
    print("last word",last_word)
    # Start from the end of last word and work backwards
    for i in range(len(last_word)):
        prefix = last_word[-(i+1):]  # Take i+1 chars from the right
        start_time = time.time()
        matching_tokens = find_matching_tokens(prefix, vocab, trie)
        end_time = time.time()
        if i == 0:  # Only print timing for first iteration
            print(f"Found {len(matching_tokens)} matching tokens in {(end_time - start_time)*1000:.2f} ms")
        
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
            test_str = last_word[:-(i+1)].encode('utf-8') + token_str
            # Re-encode the test sentence
            test_tokens = enc.encode(test_str.decode('utf-8', errors='ignore'))
            
            # Get tokens for original sentence up to the replacement point
            original_tokens = enc.encode(last_word[:-(i+1)])
            # Combine with our test token
            expected_tokens = original_tokens + test_tokens[-1:]
            # Re-encode full test string to compare
            actual_tokens = enc.encode(test_str.decode('utf-8', errors='ignore'))
            
            # Check if tokens match except for the last one we replaced
            if expected_tokens == actual_tokens:
                filtered_tokens.append(token)
        matching_tokens = filtered_tokens
        matches_by_position.append(matching_tokens)
        
    return matches_by_position

def analyze_token_combinations(sentence: str, vocab: Dict[bytes, int], trie=None) -> List[Dict]:
    """
    Analyze possible token combinations for a given sentence
    
    Args:
        sentence: Input sentence to analyze
        vocab: Dictionary mapping token bytes to token ids
        trie: Optional trie data structure for faster matching
        
    Returns:
        List of dictionaries containing position, prefix and matching tokens
    """
    right_matches = find_tokens_from_right(sentence, vocab, trie)
    combinations = []
    
    for i, matches in enumerate(right_matches):
        if len(matches) > 0:
            combinations.append({
                'position': len(sentence) - i,
                'prefix': sentence[:-(i+1)],
                'matches': matches
            })
            
    return combinations

def init_trie(vocab):
    """Initialize the trie from vocabulary if not already done"""
    global vocab_trie
    if vocab_trie is None:
        vocab_trie = build_trie_from_vocab(vocab)
    return vocab_trie

if __name__ == "__main__":
    # Initialize trie (this only happens once)
    trie = init_trie(vocab)
    
    # Visualize the trie structure
    print("\nVisualizing the trie structure (default parameters):")
    trie.print_trie()
    
    # Visualize with custom parameters
    print("\nVisualizing with custom parameters (deeper view):")
    trie.print_trie(max_depth=4, max_children=3, max_tokens=2)
    
    # Visualize specific subtree
    print("\nVisualizing subtree for prefix 'user':")
    trie.print_trie(start_prefix=b'user', max_depth=4, max_children=3, max_tokens=2)
    
    # Example usage
    test_cases = [
        # Basic word completion
        "I bought some apple",  # -> apples
        
        # Token Healing
        "https:",  # -> https://
        
        # Multiple valid completions
        "userNa",  # -> userName or userNames
        
        # Hidden subword patterns
        "We found a hidden causali",  # -> causality
        
        # Token boundary misalignment
        "He introduced an intermediar",  # -> intermediary
        
        # Complex tokenization patterns
        "indivi",  # -> indivisible or individual
    ]

    for test_sentence in test_cases:
        print(f"\nTesting: {test_sentence}")
        start_time = time.time()
        combinations = analyze_token_combinations(test_sentence, vocab, trie)
        end_time = time.time()
        print(f"Analysis completed in {(end_time - start_time)*1000:.2f} ms")
                
        print("Combinations:")
        for c in combinations:
            print(f"Pos {c['position']}: {c['prefix']} -> {c['matches'][:3]}... {len(c['matches'])} possible tokens")
            
        # Visualize trie for the end of each test case
        if test_sentence and test_sentence[-1].isalpha():
            last_char = test_sentence[-1].encode('utf-8')
            # print(f"\nVisualization for prefix ending with '{test_sentence[-1]}':")
            # trie.print_trie(start_prefix=last_char, max_depth=2)
