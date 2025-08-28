#!/usr/bin/env python3
"""
Test script for MeRNSTA memory backends integration.

Tests all three memory backends:
- Default (Ollama-based semantic embedding)
- HRRFormer (High-dimensional symbolic compression)
- VecSymR (Human-like analogical mapping)
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import vector memory adapters
from vector_memory.hrrformer_adapter import hrrformer_vectorize, get_hrr_info
from vector_memory.vecsymr_adapter import vecsymr_vectorize, get_vecsymr_info, check_vecsymr_dependencies
from vector_memory.config import get_configured_vectorizer, get_vectorizer_info, load_memory_config
from vector_memory import get_vectorizer

def test_hrrformer():
    """Test HRRFormer symbolic compression backend."""
    print("\n🧠 Testing HRRFormer Backend:")
    print("="*50)
    
    test_text = "The cat chased the mouse through the garden."
    
    try:
        vector = hrrformer_vectorize(test_text)
        print(f"✅ Input: '{test_text}'")
        print(f"✅ Vector length: {len(vector)}")
        print(f"✅ Vector sample: {vector[:5]}...")
        print(f"✅ Vector type: {type(vector[0])}")
        
        # Get backend info
        info = get_hrr_info()
        print("\n📊 HRRFormer Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ HRRFormer test failed: {e}")
        import traceback
        traceback.print_exc()

def test_vecsymr():
    """Test VecSymR analogical mapping backend."""
    print("\n🔗 Testing VecSymR Backend:")
    print("="*50)
    
    test_text = "The cat chased the mouse through the garden."
    
    # First check dependencies
    deps = check_vecsymr_dependencies()
    print("📋 Dependency Check:")
    for key, status in deps.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {key}: {status}")
    
    try:
        vector = vecsymr_vectorize(test_text)
        print(f"\n✅ Input: '{test_text}'")
        print(f"✅ Vector length: {len(vector)}")
        print(f"✅ Vector sample: {vector[:5]}...")
        print(f"✅ Vector type: {type(vector[0])}")
        
        # Get backend info
        info = get_vecsymr_info()
        print("\n📊 VecSymR Info:")
        for key, value in info.items():
            if key != 'dependencies':  # Already shown above
                print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"❌ VecSymR test failed: {e}")
        import traceback
        traceback.print_exc()

def test_default():
    """Test default Ollama-based backend."""
    print("\n🌐 Testing Default (Ollama) Backend:")
    print("="*50)
    
    test_text = "The cat chased the mouse through the garden."
    
    try:
        vectorizer = get_vectorizer('default')
        vector = vectorizer(test_text)
        print(f"✅ Input: '{test_text}'")
        print(f"✅ Vector length: {len(vector)}")
        print(f"✅ Vector sample: {vector[:5]}...")
        print(f"✅ Vector type: {type(vector[0])}")
        
    except Exception as e:
        print(f"❌ Default backend test failed: {e}")
        import traceback
        traceback.print_exc()

def test_configured_vectorizer():
    """Test the configured vectorizer based on config.yaml."""
    print("\n⚙️ Testing Configured Vectorizer:")
    print("="*50)
    
    try:
        # Show current configuration
        config = load_memory_config()
        print("📝 Current Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # Get vectorizer info
        info = get_vectorizer_info()
        print("\n📊 Vectorizer Info:")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # Test the configured vectorizer
        vectorizer = get_configured_vectorizer()
        test_text = "The cat chased the mouse through the garden."
        vector = vectorizer(test_text)
        
        print(f"\n✅ Test Input: '{test_text}'")
        print(f"✅ Vector length: {len(vector)}")
        print(f"✅ Vector sample: {vector[:5]}...")
        
    except Exception as e:
        print(f"❌ Configured vectorizer test failed: {e}")
        import traceback
        traceback.print_exc()

def test_backend_switching():
    """Test switching between different backends."""
    print("\n🔄 Testing Backend Switching:")
    print("="*50)
    
    test_text = "Knowledge is power in the digital age."
    backends = ['default', 'hrrformer', 'vecsymr']
    
    results = {}
    
    for backend in backends:
        try:
            print(f"\n🔄 Testing {backend} backend...")
            vectorizer = get_vectorizer(backend)
            vector = vectorizer(test_text)
            results[backend] = {
                'success': True,
                'vector_length': len(vector),
                'sample': vector[:3]
            }
            print(f"   ✅ Success - Vector length: {len(vector)}")
            
        except Exception as e:
            results[backend] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Failed: {e}")
    
    # Summary
    print("\n📊 Backend Switching Summary:")
    for backend, result in results.items():
        if result['success']:
            print(f"   ✅ {backend}: Length {result['vector_length']}, Sample {result['sample']}")
        else:
            print(f"   ❌ {backend}: {result['error']}")

def main():
    """Run all memory backend tests."""
    print("🧠 MeRNSTA Memory Backends Test Suite")
    print("="*70)
    print("Testing integration of HRRFormer, VecSymR, and Default backends")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run individual backend tests
    test_hrrformer()
    test_vecsymr()
    test_default()
    
    # Test configuration system
    test_configured_vectorizer()
    
    # Test backend switching
    test_backend_switching()
    
    print("\n🎯 Test Suite Complete!")
    print("="*70)
    print("\n💡 Usage Notes:")
    print("   - Configure memory.vector_backend in config.yaml")
    print("   - Supported backends: default, hrrformer, vecsymr") 
    print("   - HRRFormer: High-dimensional symbolic compression")
    print("   - VecSymR: Human-like analogical mapping (requires R)")
    print("   - Default: Standard Ollama-based semantic embedding")

if __name__ == "__main__":
    main()