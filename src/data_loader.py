"""
Data Loader Module for Time Series Forecasting Ablation Study

Multi-source data loading with automatic environment detection.
Supports Kaggle, local, and URL-based data loading.

This module provides a unified interface for loading training and test data
from different sources while maintaining consistency across environments.
"""

import os
import requests
from pathlib import Path
import polars as pl
import yaml
import logging
import re
import time
from typing import Tuple, Optional, Dict, Any

# Try to import gdown for Google Drive downloads
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    logging.warning("gdown not available. Google Drive downloads may not work for large files.")

class DataLoader:
    """
    Unified data loader with automatic environment detection.
    
    Supports three loading modes:
    1. Kaggle environment (automatic detection)
    2. Local files (relative/absolute paths)
    3. Internet download (for external access)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary or None to load from YAML
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config is None:
            config = DataLoader._load_config()
        
        # Handle nested structure
        paths_config = config.get('paths', {})
        if 'paths' in paths_config:
            # Handle double nested structure
            self.paths_config = paths_config['paths']
        else:
            self.paths_config = paths_config
        
        self.data_config = self.paths_config.get('data', {})
        
        # Environment detection
        self.env = self._detect_environment()
        self.logger.info(f"Detected environment: {self.env}")
        
        # Set paths based on environment
        self.train_path, self.test_path = self._get_data_paths()
    
    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        config_dir = Path('../config') if Path('../config').exists() else Path('config')
        
        for config_file in ['paths.yaml', 'horizons.yaml', 'models.yaml']:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config[config_file.replace('.yaml', '')] = yaml.safe_load(f)
        
        return config
    
    def _detect_environment(self) -> str:
        """
        Automatically detect the current environment.
        
        Returns:
            str: 'kaggle', 'local', or 'download'
        """
        # Check for Kaggle environment
        if os.path.exists('/kaggle/input'):
            return 'kaggle'
        
        # Check for local data files
        local_paths = self._get_local_paths()
        # Convert to absolute paths for checking
        absolute_paths = {key: Path(path).resolve() for key, path in local_paths.items()}
        if all(path.exists() for path in absolute_paths.values()):
            return 'local'
        
        # Default to download mode
        return 'download'
    
    def _get_local_paths(self) -> Dict[str, str]:
        """Get local data paths from configuration."""
        return {
            'train': self.data_config.get('local', {}).get('train', '../data/train.parquet'),
            'test': self.data_config.get('local', {}).get('test', '../data/test.parquet')
        }
    
    def _get_data_paths(self) -> Tuple[str, str]:
        """
        Get data paths based on detected environment.
        
        Returns:
            Tuple[str, str]: (train_path, test_path)
        """
        if self.env == 'kaggle':
            train_path = self.data_config.get('kaggle', {}).get('train', '/kaggle/input/competitions/ts-forecasting/train.parquet')
            test_path = self.data_config.get('kaggle', {}).get('test', '/kaggle/input/competitions/ts-forecasting/test.parquet')
            
        elif self.env == 'local':
            local_paths = self._get_local_paths()
            train_path = local_paths['train']
            test_path = local_paths['test']
            
        else:  # download
            download_dir = Path(self.data_config.get('download', {}).get('base_dir', './data'))
            # Create directory if it doesn't exist
            download_dir.mkdir(parents=True, exist_ok=True)
            train_path = str(download_dir / 'train.parquet')
            test_path = str(download_dir / 'test.parquet')
        
        return train_path, test_path
    
    def _download_from_url(self, url: str, destination: Path) -> None:
        """
        Download file from URL with progress tracking.
        
        Args:
            url: URL to download from
            destination: Local path to save file
        """
        self.logger.info(f"Downloading from {url} to {destination}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress logging (every 10MB)
                        if downloaded % (10 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            self.logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB ({progress:.1f}%)")
            
            self.logger.info(f"Successfully downloaded {destination}")
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {str(e)}")
            raise
    
    def _download_from_gdrive(self, file_id: str, destination: Path) -> None:
        """
        Download files from Google Drive using gdown library.
        
        Args:
            file_id: Google Drive file ID
            destination: Local path to save file
        """
        self.logger.info(f"Downloading from Google Drive (ID: {file_id[:8]}...) to {destination}")
        
        if not GDOWN_AVAILABLE:
            raise ImportError("gdown library not available. Install with: pip install gdown")
        
        try:
            # Create directory if it doesn't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Use gdown to download the file
            url = f"https://drive.google.com/uc?id={file_id}"
            self.logger.info(f"Using gdown to download from: {url}")
            
            gdown.download(url, str(destination), quiet=False)
            
            # Verify the file was downloaded and is valid
            if destination.exists():
                file_size_mb = destination.stat().st_size / (1024 * 1024)
                self.logger.info(f"Downloaded {destination.name}: {file_size_mb:.1f}MB")
                
                # Verify it's a valid parquet file
                if file_size_mb < 1:
                    # Check if it's an HTML error page
                    try:
                        with open(destination, 'r', errors='ignore') as f:
                            content = f.read(200)
                            if 'html' in content.lower():
                                raise ValueError("Downloaded file appears to be HTML error page")
                    except:
                        pass
                
                # Try to read as parquet to verify
                try:
                    df = pl.read_parquet(destination)
                    self.logger.info(f"Successfully verified parquet file: {df.shape}")
                except Exception as e:
                    raise ValueError(f"Downloaded file is not a valid parquet: {e}")
            else:
                raise FileNotFoundError("File was not created after download")
            
        except Exception as e:
            self.logger.error(f"Failed to download from Google Drive (ID: {file_id}): {str(e)}")
            raise
    
    def _download_data_if_needed(self) -> None:
        """Download data if in download mode and files don't exist."""
        if self.env != 'download':
            return
        
        download_config = self.data_config.get('download', {})
        train_url = download_config.get('train_url')
        test_url = download_config.get('test_url')
        train_gdrive_id = download_config.get('train_gdrive_id')
        test_gdrive_id = download_config.get('test_gdrive_id')
        download_dir = Path(download_config.get('base_dir', './data'))
        # Create directory if it doesn't exist
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for either URLs or GDrive IDs
        if not train_url and not train_gdrive_id:
            raise ValueError("Train download URL or Google Drive ID not configured in paths.yaml")
        if not test_url and not test_gdrive_id:
            raise ValueError("Test download URL or Google Drive ID not configured in paths.yaml")
        
        # Download files if they don't exist
        train_path = download_dir / 'train.parquet'
        test_path = download_dir / 'test.parquet'
        
        if not train_path.exists():
            if train_gdrive_id:
                self._download_from_gdrive(train_gdrive_id, train_path)
            else:
                self._download_from_url(train_url, train_path)
        
        if not test_path.exists():
            if test_gdrive_id:
                self._download_from_gdrive(test_gdrive_id, test_path)
            else:
                self._download_from_url(test_url, test_path)
    
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load training and test data based on environment.
        
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: (train_data, test_data)
        """
        self.logger.info(f"Loading data from {self.env} environment...")
        
        # Download data if needed
        if self.env == 'download':
            self._download_data_if_needed()
        
        try:
            # Load data
            self.logger.info(f"Loading train data from: {self.train_path}")
            train_data = pl.read_parquet(self.train_path)
            
            self.logger.info(f"Loading test data from: {self.test_path}")
            test_data = pl.read_parquet(self.test_path)
            
            self.logger.info(f"[OK] Train data shape: {train_data.shape}")
            self.logger.info(f"[OK] Test data shape: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def save_data(self, data: pl.DataFrame, filename: str, 
                 subfolder: Optional[str] = None) -> None:
        """
        Save data to results directory.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            subfolder: Optional subfolder in results directory
        """
        # Get results directory from configuration
        results_dir = Path(self.paths_config.get('results', {}).get('base_dir', './results'))
        
        if subfolder:
            output_dir = results_dir / subfolder
        else:
            output_dir = results_dir
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        output_path = output_dir / filename
        
        if filename.endswith('.csv'):
            data.write_csv(output_path)
        elif filename.endswith('.parquet'):
            data.write_parquet(output_path)
        else:
            # Default to parquet
            data.write_parquet(output_path.with_suffix('.parquet'))
        
        self.logger.info(f"Saved data to: {output_path}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the current environment and data paths.
        
        Returns:
            Dict[str, Any]: Environment information
        """
        return {
            'environment': self.env,
            'train_path': self.train_path,
            'test_path': self.test_path,
            'data_exists': {
                'train': Path(self.train_path).exists(),
                'test': Path(self.test_path).exists()
            }
        }

# Convenience function for quick usage
def load_all_data(config: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convenience function to load all data with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: (train_data, test_data)
    """
    loader = DataLoader(config)
    return loader.load_data()
