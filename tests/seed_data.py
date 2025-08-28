#!/usr/bin/env python3
"""
Test data seeding utilities for MeRNSTA tests.
Provides helper functions to create test data for various scenarios.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
from storage.memory_log import MemoryLog
from storage.memory_utils import TripletFact


def seed_spider_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with contradictory spider facts for testing.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create contradictory facts about spiders
    spider_messages = [
        ("I love spiders", "preference"),
        ("I hate spiders", "preference"),
        ("Spiders are fascinating creatures", "opinion"),
        ("Spiders are scary and dangerous", "opinion"),
        ("I keep pet spiders", "fact"),
        ("I avoid spiders at all costs", "fact"),
    ]
    
    for message, tag in spider_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract and store facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_sentiment_trajectory_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with facts that show sentiment trajectory over time.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create facts with increasing positive sentiment over time
    sentiment_messages = [
        ("I like coffee", "preference"),
        ("I really enjoy coffee", "preference"),
        ("I love coffee more than anything", "preference"),
        ("Coffee is my favorite thing in the world", "preference"),
        ("I adore coffee and drink it daily", "preference"),
    ]
    
    for i, (message, tag) in enumerate(sentiment_messages):
        # Create timestamp that increases over time
        timestamp = datetime.now() - timedelta(days=len(sentiment_messages) - i)
        
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract and store facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_belief_history_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with facts that create belief history.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create facts about a subject with belief evolution
    belief_messages = [
        ("I think AI is interesting", "belief"),
        ("AI is becoming more important", "belief"),
        ("I believe AI will change the world", "belief"),
        ("AI is the future of technology", "belief"),
        ("I'm convinced AI is essential", "belief"),
    ]
    
    for message, tag in belief_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract and store facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_reinforcement_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with facts that have sufficient frequency/confidence for reinforcement.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create facts that should trigger reinforcement
    reinforcement_messages = [
        ("My name is Alice", "identity"),
        ("I live in Seattle", "location"),
        ("I work as a developer", "profession"),
        ("I love programming", "preference"),
        ("Python is my favorite language", "preference"),
    ]
    
    # Repeat some facts to increase frequency
    for _ in range(3):  # Repeat to increase frequency
        for message, tag in reinforcement_messages:
            message_id = memory_log.log_memory("user", message, tags=[tag])
            message_ids.append(message_id)
            
            # Extract and store facts
            facts = memory_log.extract_facts(message)
            if facts:
                memory_log.store_facts(facts)
    
    return message_ids


def seed_pizza_contradictions(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with contradictory pizza facts for consolidation tests.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create contradictory pizza facts
    pizza_messages = [
        ("I love pizza", "preference"),
        ("I hate pizza", "preference"),
        ("Pizza is my favorite food", "preference"),
        ("I avoid pizza", "preference"),
        ("I eat pizza every day", "fact"),
        ("I never eat pizza", "fact"),
    ]
    
    for message, tag in pizza_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract and store facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_volatility_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with facts that show high volatility.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create facts with emotional volatility
    volatility_messages = [
        ("I love my job", "emotion"),
        ("I hate my job", "emotion"),
        ("My job is amazing", "emotion"),
        ("My job is terrible", "emotion"),
        ("I'm happy at work", "emotion"),
        ("I'm miserable at work", "emotion"),
    ]
    
    for message, tag in volatility_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract and store facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_sentiment_personality_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with facts for sentiment and personality tests.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create facts with sentiment evolution over time
    sentiment_messages = [
        ("I feel happy today", "emotion"),
        ("I'm really excited about this", "emotion"),
        ("I love this new project", "preference"),
        ("I'm feeling optimistic", "emotion"),
        ("This makes me very happy", "emotion"),
        ("I'm thrilled with the results", "emotion"),
        ("I feel great about this", "emotion"),
        ("I'm so pleased with everything", "emotion"),
    ]
    
    for message, tag in sentiment_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_belief_evolution_facts(memory_log: MemoryLog) -> List[int]:
    """
    Seed the database with facts showing belief evolution over time.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        List of message IDs created
    """
    message_ids = []
    
    # Create facts showing belief evolution
    belief_messages = [
        ("I think AI is interesting", "belief"),
        ("AI is becoming more important", "belief"),
        ("I believe AI will change the world", "belief"),
        ("AI is the future of technology", "belief"),
        ("I'm convinced AI is essential", "belief"),
        ("AI will revolutionize everything", "belief"),
        ("I'm certain AI is the key", "belief"),
    ]
    
    for message, tag in belief_messages:
        message_id = memory_log.log_memory("user", message, tags=[tag])
        message_ids.append(message_id)
        
        # Extract facts
        facts = memory_log.extract_facts(message)
        if facts:
            memory_log.store_facts(facts)
    
    return message_ids


def seed_comprehensive_test_data(memory_log: MemoryLog) -> Dict[str, List[int]]:
    """
    Seed the database with comprehensive test data for all test scenarios.
    
    Args:
        memory_log: MemoryLog instance to seed
        
    Returns:
        Dictionary mapping test scenario to message IDs
    """
    results = {}
    
    # Seed all types of test data
    results['spider_facts'] = seed_spider_facts(memory_log)
    results['sentiment_trajectory'] = seed_sentiment_trajectory_facts(memory_log)
    results['belief_history'] = seed_belief_history_facts(memory_log)
    results['reinforcement'] = seed_reinforcement_facts(memory_log)
    results['pizza_contradictions'] = seed_pizza_contradictions(memory_log)
    results['volatility'] = seed_volatility_facts(memory_log)
    results['sentiment_personality'] = seed_sentiment_personality_facts(memory_log)
    
    return results


def get_test_facts_summary(memory_log: MemoryLog) -> Dict[str, Any]:
    """
    Get a summary of facts in the database for testing.
    
    Args:
        memory_log: MemoryLog instance to analyze
        
    Returns:
        Dictionary with fact summary
    """
    facts = memory_log.get_all_facts()
    
    summary = {
        'total_facts': len(facts),
        'subjects': {},
        'contradictions': [],
        'high_confidence': [],
        'low_confidence': [],
    }
    
    # Group facts by subject
    for fact in facts:
        subject = fact.subject
        if subject not in summary['subjects']:
            summary['subjects'][subject] = []
        summary['subjects'][subject].append(fact)
        
        # Track confidence levels
        if fact.confidence > 0.8:
            summary['high_confidence'].append(fact)
        elif fact.confidence < 0.3:
            summary['low_confidence'].append(fact)
    
    # Find potential contradictions (same subject, different objects)
    for subject, subject_facts in summary['subjects'].items():
        if len(subject_facts) > 1:
            objects = [f.object for f in subject_facts]
            if len(set(objects)) > 1:
                summary['contradictions'].append({
                    'subject': subject,
                    'facts': subject_facts
                })
    
    return summary 