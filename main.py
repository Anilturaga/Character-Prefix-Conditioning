from transformers import AutoTokenizer
from typing import Dict, List, Optional
import re

def custom_encode(text: str, tokenizer) -> list:
    """
    Custom implementation of tokenizer.encode that uses the tokenizer's vocabulary
    for autocompletion purposes
    
    Args:
        text: The input string to tokenize
        tokenizer: The tokenizer object with vocabulary
        
    Returns:
        List of integer tokens suitable for autocompletion
    """
    # Get the vocabulary from the tokenizer
    vocab = tokenizer.get_vocab()
    
    # Initialize output tokens list
    tokens = []
    
    # Add BOS token if the tokenizer uses it
    if tokenizer.bos_token_id is not None:
        tokens.append(tokenizer.bos_token_id)
    
    # Simple greedy tokenization algorithm
    remaining_text = text
    while remaining_text:
        best_token = None
        best_token_id = None
        best_token_len = 0
        
        # Check all tokens in vocabulary for the best match at current position
        for token, token_id in vocab.items():
            # Skip special tokens in the vocab
            if token in tokenizer.all_special_tokens:
                continue
                
            # Convert token to actual string if it's a byte-level token
            token_str = token
            if token.startswith('Ġ'):  # Common in some tokenizers for space prefix
                token_str = ' ' + token[1:]
            elif token.startswith('##'):  # Common in BERT tokenizers
                token_str = token[2:]
                
            # Check if the current token matches the beginning of remaining text
            if remaining_text.startswith(token_str) and len(token_str) > best_token_len:
                best_token = token_str
                best_token_id = token_id
                best_token_len = len(token_str)
        
        # If no match found in vocab, use unknown token
        if best_token is None:
            tokens.append(tokenizer.unk_token_id)
            remaining_text = remaining_text[1:]  # Skip one character
        else:
            tokens.append(best_token_id)
            remaining_text = remaining_text[len(best_token):]
    
    # No EOS token for autocompletion use case
    
    return tokens

def string_to_tokens(text: str, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> list:
    """
    Convert a string to tokens using transformers tokenizer
    
    Args:
        text: The input string to tokenize
        model_name: Name of the model/tokenizer to use
        
    Returns:
        List of integer tokens
    """
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use our custom encoding function instead of tokenizer.encode
    tokens = custom_encode(text, tokenizer)
    
    return tokens

def tokens_to_string(tokens: list, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    """
    Convert tokens back to a string
    
    Args:
        tokens: List of integer tokens
        model_name: Name of the model/tokenizer to use
        
    Returns:
        Decoded string
    """
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Convert tokens back to string
    text = tokenizer.decode(tokens)
    
    return text

if __name__ == "__main__":
    text = "Hello, world!"
    tokens = string_to_tokens(text)
    print(tokens)
    text_from_tokens = tokens_to_string(tokens)
    print(text_from_tokens)
    
