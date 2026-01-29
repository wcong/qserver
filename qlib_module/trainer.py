"""
Qlib Trainer Module
Handles model training using Microsoft's Qlib framework

Supports:
- Alpha158 + LGBModel
- Alpha360 + LGBModel
- Alpha158 + Transformer
- Alpha360 + Transformer

Prediction horizons:
- 1 day
- 6 days
- 1 month (20 trading days)
"""
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Training output path
TRAIN_OUTPUT_PATH = "/app/data/train"


class FeatureSet(Enum):
    """Available feature sets."""
    ALPHA158 = "alpha158"
    ALPHA360 = "alpha360"


class ModelType(Enum):
    """Available model types."""
    LGBMODEL = "lgbmodel"
    TRANSFORMER = "transformer"


class PredictionHorizon(Enum):
    """Prediction horizons in trading days."""
    DAY_1 = 1
    DAY_6 = 6
    MONTH_1 = 20  # Approximately 1 month of trading days


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    feature_set: FeatureSet
    model_type: ModelType
    horizon: PredictionHorizon
    market: str = "csi300"
    train_start: str = "2017-01-01"
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2024-12-31"
    
    def get_model_name(self) -> str:
        """Generate model name from configuration."""
        return f"{self.feature_set.value}_{self.model_type.value}_horizon{self.horizon.value}"
    
    def get_label_expression(self) -> str:
        """Get label expression for the prediction horizon."""
        # Ref($close, -n) means the close price n days later
        # Label = future return = (future_close - current_close) / current_close
        return f"Ref($close, -{self.horizon.value})/Ref($close, -1) - 1"


class QlibTrainer:
    """Trainer class for Qlib models with multiple feature sets and model types."""
    
    # All training combinations
    TRAINING_COMBINATIONS = [
        (FeatureSet.ALPHA158, ModelType.LGBMODEL),
        (FeatureSet.ALPHA360, ModelType.LGBMODEL),
        (FeatureSet.ALPHA158, ModelType.TRANSFORMER),
        (FeatureSet.ALPHA360, ModelType.TRANSFORMER),
    ]
    
    # All prediction horizons
    PREDICTION_HORIZONS = [
        PredictionHorizon.DAY_1,
        PredictionHorizon.DAY_6,
        PredictionHorizon.MONTH_1,
    ]
    
    def __init__(self, data_path: str = None, train_output_path: str = None):
        """
        Initialize the QlibTrainer.
        
        Args:
            data_path: Path to qlib data directory
            train_output_path: Path to save training results
        """
        self.data_path = data_path or os.environ.get('QLIB_DATA_PATH', '/app/data/stock/cn_data')
        self.train_output_path = train_output_path or os.environ.get('TRAIN_OUTPUT_PATH', TRAIN_OUTPUT_PATH)
        self.training_history_file = os.path.join(self.train_output_path, 'training_history.json')
        self._qlib_initialized = False
        
        # Ensure directories exist
        os.makedirs(self.train_output_path, exist_ok=True)
        os.makedirs(os.path.join(self.train_output_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.train_output_path, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.train_output_path, 'predictions'), exist_ok=True)
    
    def _init_qlib(self):
        """Initialize Qlib with the configured data path."""
        if self._qlib_initialized:
            return
        
        try:
            import qlib
            qlib.init(provider_uri=self.data_path, region="cn")
            self._qlib_initialized = True
            logger.info(f"Qlib initialized with data path: {self.data_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize qlib: {e}")
    
    def train_all_models(self, **kwargs) -> Dict[str, Any]:
        """
        Train all model combinations for all prediction horizons.
        
        This trains:
        - Alpha158 + LGBModel (1d, 6d, 1m)
        - Alpha360 + LGBModel (1d, 6d, 1m)
        - Alpha158 + Transformer (1d, 6d, 1m)
        - Alpha360 + Transformer (1d, 6d, 1m)
        
        Returns:
            Dictionary containing all training results
        """
        self._init_qlib()
        
        results = {
            'start_time': datetime.now().isoformat(),
            'models': [],
            'status': 'running'
        }
        
        for feature_set, model_type in self.TRAINING_COMBINATIONS:
            for horizon in self.PREDICTION_HORIZONS:
                config = TrainingConfig(
                    feature_set=feature_set,
                    model_type=model_type,
                    horizon=horizon,
                    **kwargs
                )
                
                logger.info(f"Training: {config.get_model_name()}")
                try:
                    result = self.train_model(config)
                    results['models'].append(result)
                except Exception as e:
                    logger.error(f"Failed to train {config.get_model_name()}: {e}")
                    results['models'].append({
                        'name': config.get_model_name(),
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'
        
        # Save overall results
        results_file = os.path.join(self.train_output_path, 'results', 
                                     f"train_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def train_model(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Train a single model with the given configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary containing training results
        """
        self._init_qlib()
        
        training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get_model_name()
        
        result = {
            'id': training_id,
            'name': model_name,
            'feature_set': config.feature_set.value,
            'model_type': config.model_type.value,
            'horizon': config.horizon.value,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'config': {
                'market': config.market,
                'train_start': config.train_start,
                'train_end': config.train_end,
                'valid_start': config.valid_start,
                'valid_end': config.valid_end,
                'test_start': config.test_start,
                'test_end': config.test_end,
            }
        }
        
        try:
            if self._qlib_initialized:
                result = self._train_with_qlib(config, training_id)
            else:
                # Demo mode
                result['status'] = 'completed'
                result['end_time'] = datetime.now().isoformat()
                result['metrics'] = self._get_demo_metrics()
                result['message'] = 'Demo training completed (qlib not available)'
                result['model_path'] = self._save_demo_model(model_name, training_id)
            
            self._save_training_history(result)
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"Training failed for {model_name}: {e}")
            self._save_training_history(result)
        
        return result
    
    def _train_with_qlib(self, config: TrainingConfig, training_id: str) -> Dict[str, Any]:
        """
        Perform actual training with Qlib.
        
        Args:
            config: Training configuration
            training_id: Unique training ID
            
        Returns:
            Training result dictionary
        """
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord
        
        model_name = config.get_model_name()
        
        # Get data handler config based on feature set
        handler_config = self._get_handler_config(config)
        
        # Get model config based on model type
        model_config = self._get_model_config(config)
        
        # Create dataset
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": (config.train_start, config.train_end),
                    "valid": (config.valid_start, config.valid_end),
                    "test": (config.test_start, config.test_end),
                },
            },
        }
        
        dataset = init_instance_by_config(dataset_config)
        model = init_instance_by_config(model_config)
        
        # Training
        experiment_name = f"{model_name}_{training_id}"
        with R.start(experiment_name=experiment_name):
            model.fit(dataset)
            
            # Generate predictions
            rec = SignalRecord(model, dataset, recorder=R.get_recorder())
            rec.generate()
            
            # Get metrics
            metrics = R.get_recorder().list_metrics()
        
        # Save model
        model_save_path = os.path.join(
            self.train_output_path, 'models', 
            f"{model_name}_{training_id}.pkl"
        )
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'config': config,
                'training_id': training_id,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        # Save predictions
        predictions = rec.get_signal()
        pred_save_path = os.path.join(
            self.train_output_path, 'predictions',
            f"{model_name}_{training_id}_predictions.pkl"
        )
        with open(pred_save_path, 'wb') as f:
            pickle.dump(predictions, f)
        
        return {
            'id': training_id,
            'name': model_name,
            'feature_set': config.feature_set.value,
            'model_type': config.model_type.value,
            'horizon': config.horizon.value,
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'status': 'completed',
            'model_path': model_save_path,
            'predictions_path': pred_save_path,
            'metrics': metrics,
            'config': {
                'market': config.market,
                'train_start': config.train_start,
                'train_end': config.train_end,
                'valid_start': config.valid_start,
                'valid_end': config.valid_end,
                'test_start': config.test_start,
                'test_end': config.test_end,
            }
        }
    
    def _get_handler_config(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Get data handler configuration based on feature set.
        
        Args:
            config: Training configuration
            
        Returns:
            Handler configuration dictionary
        """
        label_expr = config.get_label_expression()
        
        if config.feature_set == FeatureSet.ALPHA158:
            return {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": config.train_start,
                    "end_time": config.test_end,
                    "fit_start_time": config.train_start,
                    "fit_end_time": config.train_end,
                    "instruments": config.market,
                    "label": [label_expr],
                },
            }
        elif config.feature_set == FeatureSet.ALPHA360:
            return {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": config.train_start,
                    "end_time": config.test_end,
                    "fit_start_time": config.train_start,
                    "fit_end_time": config.train_end,
                    "instruments": config.market,
                    "label": [label_expr],
                },
            }
        else:
            raise ValueError(f"Unknown feature set: {config.feature_set}")
    
    def _get_model_config(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Get model configuration based on model type.
        
        Args:
            config: Training configuration
            
        Returns:
            Model configuration dictionary
        """
        if config.model_type == ModelType.LGBMODEL:
            return {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                    "early_stopping_rounds": 200,
                    "num_boost_round": 1000,
                },
            }
        elif config.model_type == ModelType.TRANSFORMER:
            # Determine input dimension based on feature set
            d_feat = 158 if config.feature_set == FeatureSet.ALPHA158 else 360
            return {
                "class": "TransformerModel",
                "module_path": "qlib.contrib.model.pytorch_transformer",
                "kwargs": {
                    "d_feat": d_feat,
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "n_epochs": 100,
                    "lr": 0.0001,
                    "early_stop": 20,
                    "batch_size": 2048,
                    "metric": "loss",
                    "loss": "mse",
                    "GPU": 0,
                },
            }
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def _get_demo_metrics(self) -> Dict[str, float]:
        """Get demo metrics for testing without qlib."""
        import random
        return {
            'IC': round(random.uniform(0.03, 0.08), 4),
            'ICIR': round(random.uniform(0.3, 0.6), 4),
            'Rank_IC': round(random.uniform(0.04, 0.09), 4),
            'Rank_ICIR': round(random.uniform(0.35, 0.65), 4),
            'Annualized_Return': round(random.uniform(0.1, 0.3), 4),
            'Max_Drawdown': round(random.uniform(-0.2, -0.05), 4),
            'Sharpe_Ratio': round(random.uniform(1.0, 2.5), 4),
        }
    
    def _save_demo_model(self, model_name: str, training_id: str) -> str:
        """Save a demo model placeholder."""
        model_path = os.path.join(
            self.train_output_path, 'models',
            f"{model_name}_{training_id}.pkl"
        )
        with open(model_path, 'wb') as f:
            pickle.dump({
                'demo': True,
                'name': model_name,
                'training_id': training_id,
                'timestamp': datetime.now().isoformat()
            }, f)
        return model_path
    
    def load_trained_model(
        self, 
        feature_set: FeatureSet, 
        model_type: ModelType, 
        horizon: PredictionHorizon,
        training_id: str = None
    ) -> Tuple[Any, TrainingConfig]:
        """
        Load a trained model for finetuning or inference.
        
        Args:
            feature_set: Feature set used for training
            model_type: Model type
            horizon: Prediction horizon
            training_id: Specific training ID (optional, uses latest if not provided)
            
        Returns:
            Tuple of (model, config)
        """
        model_name = f"{feature_set.value}_{model_type.value}_horizon{horizon.value}"
        models_dir = os.path.join(self.train_output_path, 'models')
        
        if training_id:
            model_path = os.path.join(models_dir, f"{model_name}_{training_id}.pkl")
        else:
            # Find the latest model
            model_files = [
                f for f in os.listdir(models_dir) 
                if f.startswith(model_name) and f.endswith('.pkl')
            ]
            if not model_files:
                raise FileNotFoundError(f"No trained model found for {model_name}")
            
            # Sort by timestamp (filename contains timestamp)
            model_files.sort(reverse=True)
            model_path = os.path.join(models_dir, model_files[0])
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded model from: {model_path}")
        return data.get('model'), data.get('config')
    
    def finetune_model(
        self,
        feature_set: FeatureSet,
        model_type: ModelType,
        horizon: PredictionHorizon,
        finetune_start: str = None,
        finetune_end: str = None,
        training_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Finetune a trained model with the newest data.
        
        Args:
            feature_set: Feature set used for training
            model_type: Model type
            horizon: Prediction horizon
            finetune_start: Start date for finetuning data
            finetune_end: End date for finetuning data
            training_id: Specific training ID to finetune (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing finetuning results
        """
        self._init_qlib()
        
        # Load existing model
        try:
            model, original_config = self.load_trained_model(
                feature_set, model_type, horizon, training_id
            )
        except FileNotFoundError as e:
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Could not load model for finetuning'
            }
        
        # Set default dates if not provided
        if finetune_start is None:
            finetune_start = datetime.now().strftime('%Y-01-01')
        if finetune_end is None:
            finetune_end = datetime.now().strftime('%Y-%m-%d')
        
        finetune_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{feature_set.value}_{model_type.value}_horizon{horizon.value}"
        
        result = {
            'id': finetune_id,
            'name': f"{model_name}_finetuned",
            'original_model': model_name,
            'feature_set': feature_set.value,
            'model_type': model_type.value,
            'horizon': horizon.value,
            'finetune_start': finetune_start,
            'finetune_end': finetune_end,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
        }
        
        try:
            if self._qlib_initialized and model is not None:
                result = self._finetune_with_qlib(
                    model, feature_set, model_type, horizon,
                    finetune_start, finetune_end, finetune_id, 
                    original_config, **kwargs
                )
            else:
                # Demo mode
                result['status'] = 'completed'
                result['end_time'] = datetime.now().isoformat()
                result['metrics'] = self._get_demo_metrics()
                result['message'] = 'Demo finetuning completed (qlib not available or demo model)'
                result['model_path'] = self._save_demo_model(
                    f"{model_name}_finetuned", finetune_id
                )
            
            self._save_training_history(result)
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"Finetuning failed for {model_name}: {e}")
            self._save_training_history(result)
        
        return result
    
    def _finetune_with_qlib(
        self,
        model: Any,
        feature_set: FeatureSet,
        model_type: ModelType,
        horizon: PredictionHorizon,
        finetune_start: str,
        finetune_end: str,
        finetune_id: str,
        original_config: TrainingConfig = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform actual finetuning with Qlib.
        
        Args:
            model: Pre-trained model to finetune
            feature_set: Feature set
            model_type: Model type
            horizon: Prediction horizon
            finetune_start: Finetuning data start date
            finetune_end: Finetuning data end date
            finetune_id: Unique finetuning ID
            original_config: Original training configuration
            **kwargs: Additional parameters
            
        Returns:
            Finetuning result dictionary
        """
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord
        
        model_name = f"{feature_set.value}_{model_type.value}_horizon{horizon.value}"
        market = kwargs.get('market', original_config.market if original_config else 'csi300')
        
        # Create config for finetuning
        config = TrainingConfig(
            feature_set=feature_set,
            model_type=model_type,
            horizon=horizon,
            market=market,
            train_start=finetune_start,
            train_end=finetune_end,
            valid_start=finetune_start,
            valid_end=finetune_end,
            test_start=finetune_start,
            test_end=finetune_end,
        )
        
        # Get handler config
        handler_config = self._get_handler_config(config)
        
        # Create dataset for finetuning
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": (finetune_start, finetune_end),
                },
            },
        }
        
        dataset = init_instance_by_config(dataset_config)
        
        # Finetuning
        experiment_name = f"{model_name}_finetune_{finetune_id}"
        with R.start(experiment_name=experiment_name):
            # For LGBModel, we retrain with warm start
            # For Transformer, we continue training
            if model_type == ModelType.LGBMODEL:
                # LightGBM doesn't support true finetuning, so we retrain
                model_config = self._get_model_config(config)
                model = init_instance_by_config(model_config)
                model.fit(dataset)
            else:
                # Transformer can be finetuned
                model.fit(dataset, num_boost_round=100)  # Fewer epochs for finetuning
            
            # Generate predictions
            rec = SignalRecord(model, dataset, recorder=R.get_recorder())
            rec.generate()
            
            metrics = R.get_recorder().list_metrics()
        
        # Save finetuned model
        model_save_path = os.path.join(
            self.train_output_path, 'models',
            f"{model_name}_finetuned_{finetune_id}.pkl"
        )
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'config': config,
                'finetune_id': finetune_id,
                'timestamp': datetime.now().isoformat(),
                'finetuned': True,
                'finetune_start': finetune_start,
                'finetune_end': finetune_end,
            }, f)
        
        return {
            'id': finetune_id,
            'name': f"{model_name}_finetuned",
            'original_model': model_name,
            'feature_set': feature_set.value,
            'model_type': model_type.value,
            'horizon': horizon.value,
            'finetune_start': finetune_start,
            'finetune_end': finetune_end,
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'status': 'completed',
            'model_path': model_save_path,
            'metrics': metrics,
        }
    
    def finetune_all_models(
        self,
        finetune_start: str = None,
        finetune_end: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Finetune all trained models with the newest data.
        
        Args:
            finetune_start: Start date for finetuning data
            finetune_end: End date for finetuning data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing all finetuning results
        """
        results = {
            'start_time': datetime.now().isoformat(),
            'models': [],
            'status': 'running'
        }
        
        for feature_set, model_type in self.TRAINING_COMBINATIONS:
            for horizon in self.PREDICTION_HORIZONS:
                logger.info(f"Finetuning: {feature_set.value}_{model_type.value}_horizon{horizon.value}")
                try:
                    result = self.finetune_model(
                        feature_set=feature_set,
                        model_type=model_type,
                        horizon=horizon,
                        finetune_start=finetune_start,
                        finetune_end=finetune_end,
                        **kwargs
                    )
                    results['models'].append(result)
                except Exception as e:
                    logger.error(f"Finetuning failed: {e}")
                    results['models'].append({
                        'name': f"{feature_set.value}_{model_type.value}_horizon{horizon.value}",
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'
        
        # Save results
        results_file = os.path.join(
            self.train_output_path, 'results',
            f"finetune_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_training_history(self, result: Dict[str, Any]):
        """Save training result to history file."""
        history = self.get_training_history()
        history.append(result)
        
        # Keep only last 500 records
        history = history[-500:]
        
        with open(self.training_history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        if os.path.exists(self.training_history_file):
            try:
                with open(self.training_history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models."""
        models = []
        models_dir = os.path.join(self.train_output_path, 'models')
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(models_dir, filename)
                    
                    # Parse model info from filename
                    parts = filename.replace('.pkl', '').split('_')
                    
                    model_info = {
                        'filename': filename,
                        'path': filepath,
                        'created': datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).isoformat(),
                        'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
                    }
                    
                    # Try to extract more info
                    if len(parts) >= 3:
                        model_info['feature_set'] = parts[0]
                        model_info['model_type'] = parts[1]
                        if 'horizon' in parts[2]:
                            model_info['horizon'] = parts[2].replace('horizon', '')
                        model_info['finetuned'] = 'finetuned' in filename
                    
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x['created'], reverse=True)
    
    def get_model_predictions(
        self,
        feature_set: FeatureSet,
        model_type: ModelType,
        horizon: PredictionHorizon,
        training_id: str = None
    ) -> Any:
        """
        Get predictions from a trained model.
        
        Args:
            feature_set: Feature set
            model_type: Model type
            horizon: Prediction horizon
            training_id: Specific training ID (optional)
            
        Returns:
            Predictions data
        """
        model_name = f"{feature_set.value}_{model_type.value}_horizon{horizon.value}"
        pred_dir = os.path.join(self.train_output_path, 'predictions')
        
        if training_id:
            pred_path = os.path.join(pred_dir, f"{model_name}_{training_id}_predictions.pkl")
        else:
            # Find the latest predictions
            pred_files = [
                f for f in os.listdir(pred_dir)
                if f.startswith(model_name) and f.endswith('_predictions.pkl')
            ]
            if not pred_files:
                raise FileNotFoundError(f"No predictions found for {model_name}")
            
            pred_files.sort(reverse=True)
            pred_path = os.path.join(pred_dir, pred_files[0])
        
        with open(pred_path, 'rb') as f:
            return pickle.load(f)
    
    # Backward compatibility methods
    def train_model_legacy(self, model_type: str = "LGBModel", **kwargs) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        
        Args:
            model_type: Type of model ("LGBModel" or "Transformer")
            **kwargs: Additional parameters including feature_set and horizon
            
        Returns:
            Training result dictionary
        """
        feature_set = FeatureSet(kwargs.pop('feature_set', 'alpha158'))
        model_type_enum = ModelType.LGBMODEL if model_type == "LGBModel" else ModelType.TRANSFORMER
        horizon = PredictionHorizon(kwargs.pop('horizon', 1))
        
        config = TrainingConfig(
            feature_set=feature_set,
            model_type=model_type_enum,
            horizon=horizon,
            **kwargs
        )
        
        return self.train_model(config)
