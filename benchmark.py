import time
import tiktoken
from main import find_matching_tokens, analyze_token_combinations, init_trie, build_trie_from_vocab

def benchmark_prefix_matching(test_cases, vocab, num_trials=5):
    """
    Benchmark the performance of prefix matching with and without trie
    
    Args:
        test_cases: List of test sentences
        vocab: Dictionary mapping token bytes to token ids
        num_trials: Number of trials to run for each test
        
    Returns:
        None, prints results
    """
    # Initialize trie (one-time cost)
    trie_init_start = time.time()
    trie = init_trie(vocab)
    trie_init_time = time.time() - trie_init_start
    print(f"Trie initialization time: {trie_init_time:.4f} seconds")
    
    results = {
        'original': {'total': 0, 'avg': 0},
        'trie': {'total': 0, 'avg': 0}
    }
    
    # Test with simple prefix matching only first
    print("\n=== Simple Prefix Matching ===")
    for test_case in test_cases:
        print(f"\nTesting prefix: '{test_case}'")
        
        # Original implementation
        original_times = []
        for _ in range(num_trials):
            start_time = time.time()
            matches = find_matching_tokens(test_case, vocab)
            end_time = time.time()
            original_times.append(end_time - start_time)
        
        avg_original = sum(original_times) / num_trials
        results['original']['total'] += avg_original
        print(f"Original: {avg_original * 1000:.2f} ms (average of {num_trials} trials, {len(matches)} matches)")
        
        # Trie implementation
        trie_times = []
        for _ in range(num_trials):
            start_time = time.time()
            matches = find_matching_tokens(test_case, vocab, trie)
            end_time = time.time()
            trie_times.append(end_time - start_time)
        
        avg_trie = sum(trie_times) / num_trials
        results['trie']['total'] += avg_trie
        print(f"With Trie: {avg_trie * 1000:.2f} ms (average of {num_trials} trials, {len(matches)} matches)")
        print(f"Speedup: {avg_original / avg_trie:.2f}x")
    
    results['original']['avg'] = results['original']['total'] / len(test_cases)
    results['trie']['avg'] = results['trie']['total'] / len(test_cases)
    
    print("\n=== Overall Simple Prefix Matching Results ===")
    print(f"Original average: {results['original']['avg'] * 1000:.2f} ms per test case")
    print(f"Trie average: {results['trie']['avg'] * 1000:.2f} ms per test case")
    print(f"Overall speedup: {results['original']['avg'] / results['trie']['avg']:.2f}x")
    
    # Now test the full token combination analysis
    print("\n=== Full Token Combination Analysis ===")
    full_results = {
        'original': {'total': 0, 'avg': 0},
        'trie': {'total': 0, 'avg': 0}
    }
    
    for test_case in test_cases:
        print(f"\nTesting full sentence: '{test_case}'")
        
        # Original implementation
        start_time = time.time()
        combinations = analyze_token_combinations(test_case, vocab)
        original_time = time.time() - start_time
        full_results['original']['total'] += original_time
        print(f"Original: {original_time * 1000:.2f} ms ({len(combinations)} combinations)")
        
        # Trie implementation
        start_time = time.time()
        combinations = analyze_token_combinations(test_case, vocab, trie)
        trie_time = time.time() - start_time
        full_results['trie']['total'] += trie_time
        print(f"With Trie: {trie_time * 1000:.2f} ms ({len(combinations)} combinations)")
        print(f"Speedup: {original_time / trie_time:.2f}x")
    
    full_results['original']['avg'] = full_results['original']['total'] / len(test_cases)
    full_results['trie']['avg'] = full_results['trie']['total'] / len(test_cases)
    
    print("\n=== Overall Token Combination Analysis Results ===")
    print(f"Original average: {full_results['original']['avg'] * 1000:.2f} ms per test case")
    print(f"Trie average: {full_results['trie']['avg'] * 1000:.2f} ms per test case")
    print(f"Overall speedup: {full_results['original']['avg'] / full_results['trie']['avg']:.2f}x")
    
    # Calculate the amortized cost including trie initialization
    if len(test_cases) > 0:
        print("\n=== Amortized Cost (including trie initialization) ===")
        amortized_time_per_case = (full_results['trie']['total'] + trie_init_time) / len(test_cases)
        print(f"Amortized trie cost: {amortized_time_per_case * 1000:.2f} ms per test case")
        break_even_cases = trie_init_time / (full_results['original']['avg'] - full_results['trie']['avg'])
        print(f"Break-even point: {break_even_cases:.1f} test cases to justify trie initialization cost")

if __name__ == "__main__":
    # Get the vocabulary
    enc = tiktoken.get_encoding("cl100k_base")
    vocab = enc._mergeable_ranks
    
    # Test cases for benchmarking
    test_cases = [
        "The agreement was signed unconditiona",
        "He introduced an intermediar",
        "We found a hidden correla", 
        "I bought some apple",
        "I am an indivi",
        "I am indivi",
        "The system was designed to automat",
        "They discovered a new planet in the constella",
        "The implementation of the algorith",
        "She prepared a comprehens"
    ]
    
    # Run the benchmark
    benchmark_prefix_matching(test_cases, vocab) 