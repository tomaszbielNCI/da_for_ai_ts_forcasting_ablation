#!/usr/bin/env python3
"""
Test script for DataLoader functionality

This script tests the data loading module with different environments
and verifies that the data is loaded correctly.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from src.data_loader import DataLoader, load_all_data

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_data_loader():
    """Test DataLoader functionality"""
    print("🧪 Testing DataLoader...")
    
    # Initialize loader
    loader = DataLoader()
    
    # Get environment info
    env_info = loader.get_environment_info()
    print(f"📊 Environment Info:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    # Load data
    try:
        train_data, test_data = loader.load_data()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        
        # Show sample data
        print(f"\n📋 Train data sample:")
        print(train_data.head(3))
        
        print(f"\n📋 Test data sample:")
        print(test_data.head(3))
        
        # Test saving functionality
        print(f"\n💾 Testing save functionality...")
        loader.save_data(train_data.head(100), "test_train_sample.csv", "predictions")
        loader.save_data(test_data.head(100), "test_test_sample.parquet", "predictions")
        
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_convenience_function():
    """Test convenience function"""
    print("\n🧪 Testing convenience function...")
    
    try:
        train_data, test_data = load_all_data()
        print(f"✅ Convenience function works!")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Convenience function failed: {str(e)}")
        return False

def main():
    """Main test function"""
    setup_logging()
    
    print("Starting DataLoader tests...")
    
    # Test main functionality
    test1_passed = test_data_loader()
    
    # Test convenience function
    test2_passed = test_convenience_function()
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"  DataLoader class: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Convenience function: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! DataLoader is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")

if __name__ == "__main__":
    main()
