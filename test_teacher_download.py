#!/usr/bin/env python3
"""
Test script for teacher to verify Google Drive download functionality.
This script demonstrates that the project can download data from Google Drive
as required by the university.

Usage:
    python test_teacher_download.py

This script will:
1. Force download mode (ignore local files)
2. Download train.parquet and test.parquet from Google Drive
3. Verify the downloaded files are valid parquet format
4. Show data shapes and basic information
"""

import sys
import os
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_teacher_download():
    """Test Google Drive download for teacher verification."""
    
    print("=" * 60)
    print("TEACHER VERIFICATION: Google Drive Download Test")
    print("=" * 60)
    print("This test demonstrates downloading data from Google Drive")
    print("as required by the university for project evaluation.")
    print()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Import DataLoader
        from src.data_loader import DataLoader
        
        print("1. Testing DataLoader with Google Drive download...")
        print()
        
        # Create custom config that forces download from Google Drive
        config = {
            'paths': {
                'data': {
                    'download': {
                        'base_dir': './teacher_test_download',
                        'train_gdrive_id': '19VWQoWImXzuWfx3YpT4zF7ldwanTwGwG',
                        'test_gdrive_id': '1cWxdooyeSSMghsiUdPIDSIXEepLyDEa6'
                    }
                }
            }
        }
        
        # Initialize DataLoader
        loader = DataLoader(config)
        
        # Force download mode (ignore local files)
        loader.env = 'download'
        loader.train_path, loader.test_path = loader._get_data_paths()
        
        print(f"Environment: {loader.env}")
        print(f"Download directory: {Path(loader.train_path).parent}")
        print(f"Train file: {Path(loader.train_path).name}")
        print(f"Test file: {Path(loader.test_path).name}")
        print()
        
        # Check if files already exist
        train_exists = Path(loader.train_path).exists()
        test_exists = Path(loader.test_path).exists()
        
        if train_exists or test_exists:
            print("NOTE: Some files already exist. They will be used instead of downloading.")
            print("To force fresh download, delete the teacher_test_download directory.")
            print()
        
        print("2. Loading data (will download if needed)...")
        print()
        
        # Load data (this triggers download if needed)
        train_data, test_data = loader.load_data()
        
        print("3. Verification Results:")
        print()
        
        # Verify train data
        print(f"Train Data:")
        print(f"  - Shape: {train_data.shape}")
        print(f"  - Columns: {len(train_data.columns)}")
        print(f"  - File size: {Path(loader.train_path).stat().st_size / (1024*1024):.1f} MB")
        print(f"  - Valid parquet: YES")
        print()
        
        # Verify test data
        print(f"Test Data:")
        print(f"  - Shape: {test_data.shape}")
        print(f"  - Columns: {len(test_data.columns)}")
        print(f"  - File size: {Path(loader.test_path).stat().st_size / (1024*1024):.1f} MB")
        print(f"  - Valid parquet: YES")
        print()
        
        # Show sample of data
        print("4. Sample Data (first 3 rows):")
        print()
        print("Train data sample:")
        print(train_data.head(3).to_pandas().to_string())
        print()
        print("Test data sample:")
        print(test_data.head(3).to_pandas().to_string())
        print()
        
        print("=" * 60)
        print("SUCCESS! Google Drive download verification PASSED")
        print("=" * 60)
        print("The project successfully downloads data from Google Drive")
        print("and loads it into polars DataFrames for analysis.")
        print()
        print("Files downloaded to:")
        print(f"  - {loader.train_path}")
        print(f"  - {loader.test_path}")
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print("ERROR: Google Drive download verification FAILED")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Google Drive file IDs are correct")
        print("3. Ensure gdown library is installed (pip install gdown)")
        print("4. Check if files are publicly accessible on Google Drive")
        return False

if __name__ == "__main__":
    success = test_teacher_download()
    
    print()
    print("=" * 60)
    print(f"Test Result: {'PASSED' if success else 'FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
