#!/usr/bin/env python
"""
Cron job script for daily data download
This script is executed by cron to download the latest market data from:
https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
"""
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib_module.data_manager import DataManager

# Ensure log directory exists
os.makedirs('/app/data/logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/data/logs/download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to download data from GitHub releases."""
    logger.info("=" * 50)
    logger.info(f"Starting scheduled data download at {datetime.now().isoformat()}")
    logger.info("Data source: https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz")
    logger.info("Target path: /app/data/stock/cn_data")
    logger.info("=" * 50)
    
    try:
        # Initialize data manager with the target path
        data_manager = DataManager(data_path="/app/data/stock/cn_data")
        
        # Download data (will override existing data)
        result = data_manager.download_data(region="cn", interval="day")
        
        if result['status'] == 'completed':
            logger.info(f"Data download completed successfully: {result.get('message', '')}")
            logger.info(f"Files downloaded: {result.get('files_count', 'unknown')}")
        else:
            logger.error(f"Data download failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
            
        # Validate data
        validation = data_manager.validate_data()
        if validation['valid']:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation issues: {validation['checks']}")
            
    except Exception as e:
        logger.exception(f"Data download failed with exception: {e}")
        sys.exit(1)
    
    logger.info("Data download job completed")


if __name__ == '__main__':
    main()
