#!/usr/bin/env python3
"""
OllamaTokenizer - Wrapper for Ollama tokenization endpoints

Handles both tokenize and detokenize operations via HTTP calls to Ollama API:
- POST /api/tokenize → returns tokens for given content
- POST /api/detokenize → returns string for given token list
"""

import logging
import requests
import json
from typing import List, Optional, Union
from config.settings import get_config


class OllamaTokenizer:
    """
    Wrapper for Ollama tokenization endpoints.
    
    Provides tokenize() and detokenize() methods that communicate with
    Ollama's tokenization API endpoints.
    """
    
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "mistral"):
        """
        Initialize the tokenizer.
        
        Args:
            host: Ollama host URL
            model: Model name for tokenization
        """
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = 10
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to Ollama tokenization endpoints."""
        try:
            # Test tokenize endpoint
            test_response = requests.post(
                f"{self.host}/api/tokenize",
                json={"model": self.model, "content": "test"},
                timeout=self.timeout
            )
            test_response.raise_for_status()
            logging.info(f"✅ Ollama tokenizer connected: {self.host}")
        except Exception as e:
            logging.warning(f"⚠️ Ollama tokenizer connection failed: {e}")
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using Ollama API.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
            
        Raises:
            Exception: If tokenization fails
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            response = requests.post(
                f"{self.host}/api/tokenize",
                json={"model": self.model, "content": text},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if "tokens" in data:
                return data["tokens"]
            else:
                logging.error(f"Ollama tokenize API did not return 'tokens': {data}")
                return []
                
        except Exception as e:
            logging.error(f"Ollama tokenize failed for '{text[:50]}...': {e}")
            # Fallback: return empty list
            return []
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize tokens using Ollama API.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Detokenized text
            
        Raises:
            Exception: If detokenization fails
        """
        if not tokens or not isinstance(tokens, list):
            return ""
        
        try:
            response = requests.post(
                f"{self.host}/api/detokenize",
                json={"model": self.model, "tokens": tokens},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if "content" in data:
                return data["content"]
            else:
                logging.error(f"Ollama detokenize API did not return 'content': {data}")
                return ""
                
        except Exception as e:
            logging.error(f"Ollama detokenize failed for {len(tokens)} tokens: {e}")
            # Fallback: return empty string
            return ""
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        tokens = self.tokenize(text)
        return len(tokens)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (fallback method).
        Used when Ollama tokenizer is not available.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


# Global tokenizer instance
_tokenizer_instance = None


def get_tokenizer() -> OllamaTokenizer:
    """
    Get or create the global tokenizer instance.
    
    Returns:
        OllamaTokenizer instance
    """
    global _tokenizer_instance
    
    if _tokenizer_instance is None:
        try:
            config = get_config()
            tokenizer_config = config.get('tokenizer', {})
            host = tokenizer_config.get('host', 'http://127.0.0.1:11434')
            model = tokenizer_config.get('model', 'mistral')
            
            _tokenizer_instance = OllamaTokenizer(host=host, model=model)
        except Exception as e:
            logging.error(f"Failed to initialize OllamaTokenizer: {e}")
            # Create a fallback instance
            _tokenizer_instance = OllamaTokenizer()
    
    return _tokenizer_instance


def tokenize(text: str) -> List[int]:
    """
    Convenience function to tokenize text.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of token IDs
    """
    return get_tokenizer().tokenize(text)


def detokenize(tokens: List[int]) -> str:
    """
    Convenience function to detokenize tokens.
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Detokenized text
    """
    return get_tokenizer().detokenize(tokens)


def count_tokens(text: str) -> int:
    """
    Convenience function to count tokens.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens
    """
    return get_tokenizer().count_tokens(text) 