"""
Data Manager Module
Handles data downloading and management for Qlib
"""
import os
import json
import logging
import shutil
import stat
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Data download URL
QLIB_DATA_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"

# Default data path - relative to project root
# In Docker: /app/data/stock/cn_data
# Locally: ./data/stock/cn_data (relative to project root)
def get_default_cn_data_path():
    """Get the default cn_data path, handling both Docker and local environments."""
    # Check if running in Docker (typically /app is the workdir)
    if os.path.exists('/app') and os.getcwd().startswith('/app'):
        return '/app/data/stock/cn_data'
    
    # For local development, use path relative to the project root
    # Find project root by looking for known files/folders
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # qlib_module -> project root
    
    return str(project_root / 'data' / 'stock' / 'cn_data')

DEFAULT_CN_DATA_PATH = get_default_cn_data_path()


class DataManager:
    """Manager class for Qlib data operations."""
    
    MAX_DOWNLOAD_HISTORY = 10
    
    def __init__(self, data_path: str = None):
        """
        Initialize the DataManager.
        
        Args:
            data_path: Path to qlib data directory
        """
        self.data_path = data_path or os.environ.get('QLIB_DATA_PATH', DEFAULT_CN_DATA_PATH)
        self.status_file = os.path.join(os.path.dirname(self.data_path), 'data_status.json')
        self.history_file = os.path.join(os.path.dirname(self.data_path), 'download_history.json')
        
        # Ensure directory exists
        os.makedirs(self.data_path, exist_ok=True)
    
    def download_data(self, region: str = "cn", interval: str = "day") -> Dict[str, Any]:
        """
        Download qlib data from GitHub releases.
        Downloads from: https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
        Extracts to: /app/data/stock/cn_data (overrides existing data)
        
        Args:
            region: Data region (cn for China, us for US)
            interval: Data interval (day, 1min, etc.)
            
        Returns:
            Dictionary containing download status
        """
        result = {
            'start_time': datetime.now().isoformat(),
            'region': region,
            'interval': interval,
            'status': 'running',
            'url': QLIB_DATA_URL,
            'target_path': self.data_path
        }
        
        try:
            logger.info(f"Starting data download from: {QLIB_DATA_URL}")
            logger.info(f"Target directory: {self.data_path}")
            
            # Create a temporary file for the download
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Download the file
                logger.info("Downloading qlib_bin.tar.gz...")
                urlretrieve(QLIB_DATA_URL, tmp_path)
                logger.info(f"Download completed. File size: {os.path.getsize(tmp_path)} bytes")
                
                # Remove existing data directory to override
                if os.path.exists(self.data_path):
                    logger.info(f"Removing existing data at: {self.data_path}")
                    shutil.rmtree(self.data_path)
                
                # Create parent directory
                os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                
                # Extract the tar.gz file
                logger.info(f"Extracting to: {self.data_path}")
                
                # Use a temporary extraction directory
                extract_tmp = os.path.join(os.path.dirname(self.data_path), '_extract_tmp')
                if os.path.exists(extract_tmp):
                    shutil.rmtree(extract_tmp)
                os.makedirs(extract_tmp, exist_ok=True)
                
                with tarfile.open(tmp_path, 'r:gz') as tar:
                    tar.extractall(path=extract_tmp)
                
                # Check if extraction created a qlib_bin subfolder
                extracted_items = os.listdir(extract_tmp)
                if 'qlib_bin' in extracted_items:
                    # Move contents FROM INSIDE qlib_bin to cn_data
                    qlib_bin_path = os.path.join(extract_tmp, 'qlib_bin')
                    os.makedirs(self.data_path, exist_ok=True)
                    for item in os.listdir(qlib_bin_path):
                        src = os.path.join(qlib_bin_path, item)
                        dst = os.path.join(self.data_path, item)
                        shutil.move(src, dst)
                else:
                    # Move contents directly to cn_data
                    shutil.move(extract_tmp, self.data_path)
                
                # Clean up temp extraction folder
                if os.path.exists(extract_tmp):
                    shutil.rmtree(extract_tmp)
                
                
                
                # Make data readable for everyone (chmod -R a+r)
                self._make_readable(self.data_path)
                
                result['status'] = 'completed'
                result['message'] = 'Data downloaded and extracted successfully'
                result['end_time'] = datetime.now().isoformat()
                
                # Count files
                file_count = sum(len(files) for _, _, files in os.walk(self.data_path))
                result['files_count'] = file_count
                logger.info(f"Extraction completed. Total files: {file_count}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    logger.info("Cleaned up temporary download file")
            
        except Exception as e:
            result['status'] = 'failed'
            result['message'] = str(e)
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"Data download failed: {e}")
        
        # Update status file
        self._update_status(result)
        
        return result
    
    def _make_readable(self, path: str):
        """
        Make all files and directories readable for everyone.
        
        Args:
            path: Path to make readable recursively
        """
        logger.info(f"Setting read permissions for: {path}")
        
        for root, dirs, files in os.walk(path):
            # Make directories readable and executable for everyone
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    current_mode = os.stat(dir_path).st_mode
                    # Add read and execute for all (a+rx)
                    new_mode = current_mode | stat.S_IROTH | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP
                    os.chmod(dir_path, new_mode)
                except Exception as e:
                    logger.warning(f"Failed to set permissions on {dir_path}: {e}")
            
            # Make files readable for everyone
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    current_mode = os.stat(file_path).st_mode
                    # Add read for all (a+r)
                    new_mode = current_mode | stat.S_IROTH | stat.S_IRGRP
                    os.chmod(file_path, new_mode)
                except Exception as e:
                    logger.warning(f"Failed to set permissions on {file_path}: {e}")
        
        # Also set permissions on the root path itself
        try:
            current_mode = os.stat(path).st_mode
            new_mode = current_mode | stat.S_IROTH | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP
            os.chmod(path, new_mode)
        except Exception as e:
            logger.warning(f"Failed to set permissions on {path}: {e}")
        
        logger.info("Permissions set successfully")
    
    def download_cn1500_data(self) -> Dict[str, Any]:
        """
        Download Chinese CSI 1500 constituent data.
        
        Returns:
            Dictionary containing download status
        """
        return self.download_data(region="cn", interval="day")
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Get current data status.
        
        Returns:
            Dictionary containing data status information
        """
        status = {
            'status': 'unknown',
            'data_path': self.data_path,
            'exists': os.path.exists(self.data_path),
            'files_count': 0,
            'record_count': 0
        }
        
        # Check if data directory exists and has files
        if os.path.exists(self.data_path):
            files = []
            for root, dirs, filenames in os.walk(self.data_path):
                files.extend(filenames)
            
            status['files_count'] = len(files)
            status['status'] = 'available' if files else 'empty'
            
            # Try to get more detailed status
            if os.path.exists(self.status_file):
                try:
                    with open(self.status_file, 'r') as f:
                        saved_status = json.load(f)
                        status.update(saved_status)
                except (json.JSONDecodeError, IOError):
                    pass
        else:
            status['status'] = 'not_initialized'
        
        return status
    
    def get_last_update_time(self) -> Optional[str]:
        """
        Get the last data update time.
        
        Returns:
            ISO format timestamp or None
        """
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    return status.get('end_time')
            except (json.JSONDecodeError, IOError):
                pass
        return None
    
    def _update_status(self, status: Dict[str, Any]):
        """Update status file with latest information."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to update status file: {e}")
        
        # Also update download history
        self._add_to_history(status)
    
    def _add_to_history(self, record: Dict[str, Any]):
        """Add a download record to history, keeping only the latest MAX_DOWNLOAD_HISTORY entries."""
        history = self.get_download_history()
        history.insert(0, record)  # Add to beginning
        history = history[:self.MAX_DOWNLOAD_HISTORY]  # Keep only latest 10
        
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to update history file: {e}")
    
    def get_download_history(self) -> list:
        """
        Get download history (latest 10 downloads).
        
        Returns:
            List of download records
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the downloaded data.
        
        Returns:
            Dictionary containing validation results
        """
        result = {
            'valid': False,
            'checks': []
        }
        
        # Check if data directory exists
        if not os.path.exists(self.data_path):
            result['checks'].append({
                'name': 'data_directory',
                'passed': False,
                'message': 'Data directory does not exist'
            })
            return result
        
        result['checks'].append({
            'name': 'data_directory',
            'passed': True,
            'message': 'Data directory exists'
        })
        
        # Check for required files/directories
        required_paths = ['calendars', 'instruments', 'features']
        for path in required_paths:
            full_path = os.path.join(self.data_path, path)
            exists = os.path.exists(full_path)
            result['checks'].append({
                'name': path,
                'passed': exists,
                'message': f'{path} {"exists" if exists else "missing"}'
            })
        
        # Check overall validity
        result['valid'] = all(check['passed'] for check in result['checks'])
        
        return result
