import html
import re
from typing import Optional

# Length limits
MAX_TEXT_LENGTH = 10000
MAX_FACT_LENGTH = 5000
MAX_QUERY_LENGTH = 1000

# XSS and dangerous content patterns
DANGEROUS_PATTERNS = [
    re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # inline event handlers
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"<iframe.*?>", re.IGNORECASE),
    re.compile(r"<object.*?>", re.IGNORECASE),
    re.compile(r"vbscript:", re.IGNORECASE),
    re.compile(r"data:text/html", re.IGNORECASE),
]


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input text by removing dangerous patterns and enforcing length limits.

    Args:
        text: Input text to sanitize
        max_length: Optional custom length limit (defaults to MAX_TEXT_LENGTH)

    Returns:
        Sanitized text

    Raises:
        ValueError: If text exceeds length limit or contains dangerous content
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Strip whitespace
    text = text.strip()

    # Check length
    limit = max_length or MAX_TEXT_LENGTH
    if len(text) > limit:
        raise ValueError(f"Input exceeds maximum allowed length of {limit} characters")

    # Remove dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(text):
            text = pattern.sub("", text)

    # HTML escape to prevent XSS
    text = html.escape(text)

    return text


def sanitize_fact(fact: str) -> str:
    """Sanitize a fact string with appropriate length limits."""
    return sanitize_text(fact, MAX_FACT_LENGTH)


def sanitize_query(query: str) -> str:
    """Sanitize a search query with appropriate length limits."""
    return sanitize_text(query, MAX_QUERY_LENGTH)


def validate_safe_input(text: str) -> bool:
    """
    Validate that input is safe without modifying it.

    Returns:
        True if input is safe, False otherwise
    """
    if not isinstance(text, str):
        return False
    
    # Check length
    if len(text) > MAX_QUERY_LENGTH:
        return False
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(text):
            return False
    
    return True
