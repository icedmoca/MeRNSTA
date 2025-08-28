#!/usr/bin/env python3
"""
Enterprise-grade test to enforce no-hardcoding policy
"""

import sys
import os
import ast
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    VOLATILITY_THRESHOLDS,
    PROMPT_FORMAT,
    DEFAULT_ENTITY_CATEGORIES,
    CONFIDENCE_THRESHOLDS,
    VOLATILITY_ICONS,
    CONFIDENCE_ICONS
)

def test_no_inline_thresholds():
    """Test that no hardcoded thresholds, icons, or markers exist in the codebase"""
    print("\n🔍 Testing for hardcoded values...")
    
    files_to_check = [
        "storage/memory_log.py",
        "storage/memory_utils.py", 
        "loop/conversation.py",
        "dashboard.py"
    ]
    
    # Patterns that indicate hardcoded values (should not exist)
    bad_patterns = [
        r'0\.9',  # Hardcoded confidence threshold
        r'0\.7',  # Hardcoded confidence threshold
        r'0\.8',  # Hardcoded volatility threshold
        r'0\.3',  # Hardcoded volatility threshold
        r'🔥',    # Hardcoded icon
        r'⚡',    # Hardcoded icon
        r'✅',    # Hardcoded icon
        r'\[BEGIN MEMORY\]',  # Hardcoded marker
        r'\[END MEMORY\]'     # Hardcoded marker
    ]
    
    violations = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        for pattern in bad_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"Hardcoded pattern '{pattern}' found in {file_path}")
    
    if violations:
        print("❌ Hardcoded values found:")
        for violation in violations:
            print(f"   {violation}")
        assert False, f"Found {len(violations)} hardcoded value violations"
    else:
        print("✅ No hardcoded thresholds, icons, or markers found")

def test_config_usage():
    """Test that configuration values are properly imported and used"""
    print("\n🔧 Testing configuration usage...")
    
    # Test that all required config values exist
    required_configs = [
        "VOLATILITY_THRESHOLDS",
        "PROMPT_FORMAT", 
        "DEFAULT_ENTITY_CATEGORIES",
        "CONFIDENCE_THRESHOLDS",
        "VOLATILITY_ICONS",
        "CONFIDENCE_ICONS"
    ]
    
    for config_name in required_configs:
        if config_name in globals():
            print(f"✅ {config_name} available")
        else:
            print(f"❌ {config_name} missing")
            assert False, f"Missing required config: {config_name}"
    
    # Test that config values are properly structured
    assert isinstance(VOLATILITY_THRESHOLDS, dict), "VOLATILITY_THRESHOLDS should be a dict"
    
    assert isinstance(PROMPT_FORMAT, dict), "PROMPT_FORMAT should be a dict"
    
    assert isinstance(DEFAULT_ENTITY_CATEGORIES, dict), "DEFAULT_ENTITY_CATEGORIES should be a dict"
    
    print("✅ All configuration values properly structured")

def test_formatter_usage():
    """Test that formatters are used instead of inline logic"""
    print("\n🎨 Testing formatter usage...")
    
    files_to_check = [
        "storage/memory_log.py",
        "storage/memory_utils.py",
        "loop/conversation.py",
        "dashboard.py"
    ]
    
    # Patterns that indicate inline formatting (should not exist)
    bad_patterns = [
        r'if.*confidence.*>=.*0\.9',
        r'if.*confidence.*>=.*0\.7',
        r'if.*volatility.*>.*0\.8',
        r'if.*volatility.*<=.*0\.3',
        r'🔥.*volatility',
        r'⚡.*volatility', 
        r'✅.*confidence',
        r'\[BEGIN MEMORY\]',
        r'\[END MEMORY\]'
    ]
    
    violations = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        for pattern in bad_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"Inline formatting pattern '{pattern}' found in {file_path}")
    
    if violations:
        print("❌ Inline formatting violations found:")
        for violation in violations:
            print(f"   {violation}")
        assert False, f"Found {len(violations)} inline formatting violations"
    else:
        print("✅ No inline formatting patterns found")

def test_fact_object_interface():
    """Test that fact objects use the clean interface"""
    print("\n🧠 Testing fact object interface...")
    
    # Check that format_fact_line(fact) is used
    files_to_check = [
        "storage/memory_log.py",
        "storage/memory_utils.py",
        "dashboard.py"
    ]
    
    good_patterns = [
        r'format_fact_line\(fact\)',
        r'format_fact_display\(fact\)'
    ]
    
    violations = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check for old-style calls
        old_patterns = [
            r'format_fact_line\(.*,.*,.*\)',  # Old 3-parameter style
            r'get_volatility_icon\(.*\)',     # Direct icon calls
            r'get_confidence_icon\(.*\)'       # Direct icon calls
        ]
        
        for pattern in old_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"Old-style formatting '{pattern}' found in {file_path}")
    
    if violations:
        print("❌ Old-style formatting violations found:")
        for violation in violations:
            print(f"   {violation}")
        assert False, f"Found {len(violations)} old-style formatting violations"
    else:
        print("✅ Clean fact object interface used everywhere")

def test_import_structure():
    """Test that proper imports are used"""
    print("\n📦 Testing import structure...")
    
    files_to_check = [
        "storage/memory_log.py",
        "storage/memory_utils.py",
        "loop/conversation.py",
        "dashboard.py"
    ]
    
    required_imports = [
        "from config.settings import",
        "from storage.formatters import"
    ]
    
    violations = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        for required_import in required_imports:
            if required_import not in content:
                violations.append(f"Missing import '{required_import}' in {file_path}")
    
    if violations:
        print("❌ Import violations found:")
        for violation in violations:
            print(f"   {violation}")
        assert False, f"Found {len(violations)} import violations"
    else:
        print("✅ Proper imports used everywhere")

def test_config_completeness():
    """Test that all config values are complete and valid"""
    print("\n⚙️ Testing config completeness...")
    
    # Test volatility thresholds
    required_volatility_keys = ["stable", "medium", "high", "clarification"]
    for key in required_volatility_keys:
        if key not in VOLATILITY_THRESHOLDS:
            print(f"❌ Missing volatility threshold: {key}")
            assert False, f"Missing volatility threshold: {key}"
    
    # Test prompt format
    required_prompt_keys = ["begin_memory", "end_memory", "clarification", "no_memory", "volatility_warning"]
    for key in required_prompt_keys:
        if key not in PROMPT_FORMAT:
            print(f"❌ Missing prompt format: {key}")
            assert False, f"Missing prompt format: {key}"
    
    # Test entity categories
    if len(DEFAULT_ENTITY_CATEGORIES) < 10:
        print(f"❌ Too few entity categories: {len(DEFAULT_ENTITY_CATEGORIES)}")
        assert False, f"Too few entity categories: {len(DEFAULT_ENTITY_CATEGORIES)}"
    
    # Test icons
    if len(VOLATILITY_ICONS) < 3:
        print(f"❌ Too few volatility icons: {len(VOLATILITY_ICONS)}")
        assert False, f"Too few volatility icons: {len(VOLATILITY_ICONS)}"
    
    if len(CONFIDENCE_ICONS) < 3:
        print(f"❌ Too few confidence icons: {len(CONFIDENCE_ICONS)}")
        assert False, f"Too few confidence icons: {len(CONFIDENCE_ICONS)}"
    
    print("✅ All configuration values complete and valid")

def test_formatter_functions():
    """Test that formatter functions exist and work correctly"""
    print("\n🎨 Testing formatter functions...")
    
    # Test that formatter functions exist
    from storage.formatters import (
        format_fact_line, format_fact_display, 
        get_volatility_icon, get_confidence_icon
    )
    
    # Test with sample data
    from storage.memory_utils import TripletFact
    
    sample_fact = TripletFact(
        id=1, subject="user", predicate="likes", object="coffee",
        source_message_id=1, timestamp="2024-01-01", frequency=1
    )
    
    # Test formatter functions don't crash
    try:
        formatted_line = format_fact_line(sample_fact)
        assert isinstance(formatted_line, str), "format_fact_line should return a string"
        
        formatted_display = format_fact_display(sample_fact)
        assert isinstance(formatted_display, str), "format_fact_display should return a string"
        
        volatility_icon = get_volatility_icon(0.5)
        assert isinstance(volatility_icon, str), "get_volatility_icon should return a string"
        
        confidence_icon = get_confidence_icon(0.8)
        assert isinstance(confidence_icon, str), "get_confidence_icon should return a string"
        
        print("✅ All formatter functions work correctly")
        
    except Exception as e:
        print(f"❌ Formatter function error: {e}")
        assert False, f"Formatter function error: {e}"

def test_config_validation():
    """Test that config values are within valid ranges"""
    print("\n🔍 Testing config validation...")
    
    # Test volatility thresholds are between 0 and 1
    for key, value in VOLATILITY_THRESHOLDS.items():
        assert 0.0 <= value <= 1.0, f"Volatility threshold {key}={value} should be between 0 and 1"
    
    # Test confidence thresholds are between 0 and 1
    for key, value in CONFIDENCE_THRESHOLDS.items():
        assert 0.0 <= value <= 1.0, f"Confidence threshold {key}={value} should be between 0 and 1"
    
    # Test that thresholds are in ascending order
    volatility_values = list(VOLATILITY_THRESHOLDS.values())
    assert volatility_values == sorted(volatility_values), "Volatility thresholds should be in ascending order"
    
    confidence_values = list(CONFIDENCE_THRESHOLDS.values())
    assert confidence_values == sorted(confidence_values), "Confidence thresholds should be in ascending order"
    
    print("✅ All config values are valid")

def main():
    """Run all hardcoding policy tests"""
    print("🛡️ Enterprise-Grade No-Hardcoding Policy Test")
    print("=" * 60)
    
    tests = [
        test_no_inline_thresholds,
        test_config_usage,
        test_formatter_usage,
        test_fact_object_interface,
        test_import_structure,
        test_config_completeness,
        test_formatter_functions,
        test_config_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test {test.__name__} failed with assertion error: {e}")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Codebase follows enterprise-grade no-hardcoding policy.")
        print("\n✅ You can safely:")
        print("   • Add new fact types by updating FACT_PATTERNS")
        print("   • Tune behavior by modifying config/settings.py")
        print("   • Inject memory system into any new agent")
        print("   • Extend without fear of hidden hardcoding")
        assert True, "Test completed successfully"
    else:
        print("❌ Some tests failed. Please fix hardcoding violations before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 