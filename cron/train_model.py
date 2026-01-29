#!/usr/bin/env python
"""
Cron job script for daily model training
This script is executed by cron to train and finetune models with latest data

Trains the following model combinations:
- Alpha158 + LGBModel (1d, 6d, 1m horizons)
- Alpha360 + LGBModel (1d, 6d, 1m horizons)
- Alpha158 + Transformer (1d, 6d, 1m horizons)
- Alpha360 + Transformer (1d, 6d, 1m horizons)

Results are saved to /app/data/train/
"""
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib_module.trainer import QlibTrainer, FeatureSet, ModelType, PredictionHorizon
from qlib_module.data_manager import DataManager

# Ensure log directory exists
os.makedirs('/app/data/logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/data/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to train and finetune models."""
    logger.info("=" * 60)
    logger.info(f"Starting scheduled model training at {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        # Check if data is available
        data_manager = DataManager()
        data_status = data_manager.get_data_status()
        
        if data_status['status'] != 'available':
            logger.warning(f"Data not available: {data_status['status']}")
            logger.info("Attempting to download data first...")
            download_result = data_manager.download_data()
            if download_result['status'] != 'completed':
                logger.error("Data download failed, cannot proceed with training")
                sys.exit(1)
        
        # Initialize trainer
        trainer = QlibTrainer()
        
        # Check if models already exist for finetuning
        existing_models = trainer.list_models()
        
        if existing_models:
            logger.info(f"Found {len(existing_models)} existing models, performing finetuning...")
            
            # Finetune all models with the newest data
            result = trainer.finetune_all_models(
                finetune_start=datetime.now().strftime('%Y-01-01'),
                finetune_end=datetime.now().strftime('%Y-%m-%d')
            )
            
            logger.info("Finetuning Summary:")
            for model_result in result.get('models', []):
                status = model_result.get('status', 'unknown')
                name = model_result.get('name', 'unknown')
                if status == 'completed':
                    logger.info(f"  ✓ {name} - Success")
                else:
                    logger.error(f"  ✗ {name} - Failed: {model_result.get('error', 'Unknown')}")
        else:
            logger.info("No existing models found, training new models...")
            
            # Train all model combinations
            result = trainer.train_all_models()
            
            logger.info("Training Summary:")
            for model_result in result.get('models', []):
                status = model_result.get('status', 'unknown')
                name = model_result.get('name', 'unknown')
                if status == 'completed':
                    logger.info(f"  ✓ {name} - Success")
                    if 'metrics' in model_result:
                        logger.info(f"    Metrics: {model_result['metrics']}")
                else:
                    logger.error(f"  ✗ {name} - Failed: {model_result.get('error', 'Unknown')}")
        
        # Summary
        completed = sum(1 for m in result.get('models', []) if m.get('status') == 'completed')
        total = len(result.get('models', []))
        logger.info(f"Completed {completed}/{total} model training/finetuning tasks")
        
        if completed < total:
            logger.warning(f"{total - completed} tasks failed")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Model training/finetuning failed with exception: {e}")
        sys.exit(1)
    
    logger.info("Model training job completed successfully")


if __name__ == '__main__':
    main()
