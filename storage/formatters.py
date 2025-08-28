#!/usr/bin/env python3
"""
Centralized formatters for MeRNSTA memory system
Eliminates hardcoded formatting logic across modules
"""

from typing import List

from config.settings import (CONFIDENCE_ICONS, CONFIDENCE_THRESHOLDS,
                             PROMPT_FORMAT, VOLATILITY_ICONS,
                             VOLATILITY_THRESHOLDS)

# Contradiction icons for different levels
CONTRADICTION_ICONS = {
    "high": "⚠️",  # High contradiction
    "medium": "⚡",  # Medium contradiction
    "low": "✅",  # Low/no contradiction
    "stable": "✅",  # Stable fact
}


def get_contradiction_icon(contradiction_score: float) -> str:
    """
    Get contradiction icon based on contradiction score

    Args:
        contradiction_score: Contradiction score (0.0 to 1.0)

    Returns:
        Icon string (⚠️, ⚡, or ✅)
    """
    if contradiction_score > 0.7:
        return CONTRADICTION_ICONS["high"]
    elif contradiction_score > 0.3:
        return CONTRADICTION_ICONS["medium"]
    else:
        return CONTRADICTION_ICONS["low"]


def get_volatility_icon(volatility: float) -> str:
    """
    Get volatility icon based on volatility score

    Args:
        volatility: Volatility score

    Returns:
        Icon string (🔥, ⚡, or ✅)
    """
    if volatility > VOLATILITY_THRESHOLDS["medium"]:
        return VOLATILITY_ICONS["high"]
    elif volatility > VOLATILITY_THRESHOLDS["stable"]:
        return VOLATILITY_ICONS["medium"]
    else:
        return VOLATILITY_ICONS["stable"]


def get_confidence_icon(confidence: float) -> str:
    """
    Get confidence icon based on confidence score

    Args:
        confidence: Confidence score

    Returns:
        Icon string (✅, ⚠️, or ❓)
    """
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return CONFIDENCE_ICONS["high"]
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return CONFIDENCE_ICONS["medium"]
    else:
        return CONFIDENCE_ICONS["low"]


def get_volatility_level(volatility: float) -> str:
    """
    Get volatility level description

    Args:
        volatility: Volatility score

    Returns:
        Level description ("high", "medium", or "stable")
    """
    if volatility > VOLATILITY_THRESHOLDS["medium"]:
        return "high"
    elif volatility > VOLATILITY_THRESHOLDS["stable"]:
        return "medium"
    else:
        return "stable"


def format_triplet_fact(fact) -> str:
    """
    Format a TripletFact object with contradiction and volatility indicators

    Args:
        fact: TripletFact object with subject, predicate, object, contradiction_score, volatility_score

    Returns:
        Formatted triplet fact line with indicators
    """
    # Create the basic triplet
    triplet = f"{fact.subject} {fact.predicate} {fact.object}"

    # Add frequency info
    frequency_info = f" (seen {fact.frequency}×)"

    # Add contradiction indicator if there are contradictions
    contradiction_icon = ""
    if hasattr(fact, "contradiction_score") and fact.contradiction_score > 0:
        contradiction_icon = f" {get_contradiction_icon(fact.contradiction_score)}"

    # Add volatility indicator if volatile
    volatility_icon = ""
    if hasattr(fact, "volatility_score") and fact.volatility_score > 0:
        volatility_icon = f" {get_volatility_icon(fact.volatility_score)}"

    return f"{triplet}{frequency_info}{contradiction_icon}{volatility_icon}"


def format_fact_line(fact) -> str:
    """
    Format a single fact line with confidence and volatility indicators.

    Args:
        fact: TripletFact, MemoryFact object or dict with fact data

    Returns:
        Formatted fact line string
    """
    if hasattr(fact, "subject"):  # TripletFact object
        subject = fact.subject
        predicate = fact.predicate
        object_ = fact.object
        confidence = getattr(fact, "confidence", 1.0)
        volatility = getattr(fact, "volatility_score", 0.0)
    elif hasattr(fact, "value"):  # MemoryFact object
        # For MemoryFact, use the value directly
        fact_text = fact.value
        confidence = getattr(fact, "confidence", 1.0)
        volatility = getattr(fact, "volatility", 0.0)

        # Get confidence and volatility icons
        conf_icon = get_confidence_icon(confidence)
        vol_icon = get_volatility_icon(volatility)

        # Add confidence and volatility indicators using config thresholds
        indicators = []
        if confidence < CONFIDENCE_THRESHOLDS["medium"]:
            indicators.append(f"{conf_icon} {confidence:.2f}")
        if volatility > VOLATILITY_THRESHOLDS["medium"]:
            indicators.append(f"{vol_icon} {volatility:.2f}")

        if indicators:
            return f"{fact_text} ({', '.join(indicators)})"
        else:
            return fact_text
    else:  # Dict or tuple
        subject = fact.get(
            "subject", fact[0] if isinstance(fact, (list, tuple)) else ""
        )
        predicate = fact.get(
            "predicate", fact[1] if isinstance(fact, (list, tuple)) else ""
        )
        object_ = fact.get("object", fact[2] if isinstance(fact, (list, tuple)) else "")
        confidence = fact.get("confidence", 1.0)
        volatility = fact.get("volatility_score", 0.0)

    # Get confidence and volatility icons
    conf_icon = get_confidence_icon(confidence)
    vol_icon = get_volatility_icon(volatility)

    # Format the fact
    fact_text = f"{subject} {predicate} {object_}"

    # Add confidence and volatility indicators using config thresholds
    indicators = []
    if confidence < CONFIDENCE_THRESHOLDS["medium"]:
        indicators.append(f"{conf_icon} {confidence:.2f}")
    if volatility > VOLATILITY_THRESHOLDS["medium"]:
        indicators.append(f"{vol_icon} {volatility:.2f}")

    if indicators:
        return f"{fact_text} ({', '.join(indicators)})"
    else:
        return fact_text


def format_fact_display(fact) -> str:
    """
    Format a fact for display with confidence score included

    Args:
        fact: Fact object with confidence, volatility attributes

    Returns:
        Formatted fact line with confidence score
    """
    confidence_icon = get_confidence_icon(fact.confidence)
    # Handle both volatility and volatility_score attributes
    volatility_value = getattr(
        fact, "volatility_score", getattr(fact, "volatility", 0.0)
    )
    volatility_icon = get_volatility_icon(volatility_value)

    # Add volatility info if volatility > 0
    volatility_info = (
        f" (volatility: {volatility_value:.2f})" if volatility_value > 0 else ""
    )

    # Handle different fact types
    if hasattr(fact, "value"):  # MemoryFact
        fact_text = fact.value
    elif hasattr(fact, "subject"):  # TripletFact
        fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
    else:
        fact_text = str(fact)

    return f"{confidence_icon} {fact_text} (confidence: {fact.confidence:.2f}){volatility_info}{volatility_icon}"


def format_memory_section(facts: List, max_tokens: int = 512) -> str:
    """
    Format memory section with BEGIN/END markers

    Args:
        facts: List of formatted fact lines
        max_tokens: Maximum tokens to include

    Returns:
        Complete memory section string
    """
    if not facts:
        return PROMPT_FORMAT["no_memory"]

    # Join facts with newlines
    facts_text = "\n".join(facts)

    return (
        f"{PROMPT_FORMAT['begin_memory']}\n{facts_text}\n{PROMPT_FORMAT['end_memory']}"
    )


def format_clarification_message(high_volatility_facts: List[str]) -> str:
    """
    Format clarification message for high volatility facts

    Args:
        high_volatility_facts: List of high volatility fact descriptions

    Returns:
        Formatted clarification message
    """
    if not high_volatility_facts:
        return ""

    clarification = f"\n\n{PROMPT_FORMAT['clarification']}"
    for fact in high_volatility_facts:
        clarification += f"\n• {fact}"

    return clarification


def is_high_volatility(volatility: float) -> bool:
    """
    Check if volatility is above clarification threshold

    Args:
        volatility: Volatility score

    Returns:
        True if volatility is above threshold
    """
    return volatility > VOLATILITY_THRESHOLDS["clarification"]
