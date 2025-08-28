#!/usr/bin/env python3
"""
Prompt budgeting utilities using Ollama's tokenizer when available.

Functions:
- truncate_by_tokens(text, budget): Truncate text to fit within a token budget.
- budget_join(parts, budget, sep): Join parts ensuring the total fits within the budget.
"""

from typing import List


def _get_tokenizer():
    try:
        from utils.ollama_tokenizer import get_tokenizer
        return get_tokenizer()
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Rough estimate: 1 token ≈ 4 chars
    return max(0, len(text) // 4)


def truncate_by_tokens(text: str, budget: int) -> str:
    """Truncate text to fit the given token budget using tokenizer if available."""
    if not text or budget <= 0:
        return ""

    tok = _get_tokenizer()
    if tok is None:
        # Fallback: approximate trim by characters
        approx_ratio = 4
        return text[: budget * approx_ratio]

    try:
        tokens = tok.tokenize(text)
        if len(tokens) <= budget:
            return text
        # Detokenize the first `budget` tokens
        return tok.detokenize(tokens[:budget])
    except Exception:
        approx_ratio = 4
        return text[: budget * approx_ratio]


def budget_join(parts: List[str], budget: int, sep: str = "\n") -> str:
    """Join parts within the token budget, truncating the last part if needed."""
    if budget <= 0 or not parts:
        return ""

    tok = _get_tokenizer()
    result_parts: List[str] = []
    used = 0

    for idx, part in enumerate(parts):
        if not part:
            continue
        # tokens for separator (skip before first)
        sep_tokens = 0
        if idx > 0:
            sep_tokens = _estimate_tokens(sep) if tok is None else len(tok.tokenize(sep))

        part_tokens = _estimate_tokens(part) if tok is None else len(tok.tokenize(part))
        if used + sep_tokens + part_tokens <= budget:
            if idx > 0:
                result_parts.append(sep)
            result_parts.append(part)
            used += sep_tokens + part_tokens
        else:
            # fit truncated part
            remaining = max(0, budget - used - sep_tokens)
            if remaining > 0:
                if idx > 0:
                    result_parts.append(sep)
                result_parts.append(truncate_by_tokens(part, remaining))
            break

    return "".join(result_parts)


