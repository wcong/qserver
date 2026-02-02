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
import traceback
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
            logger.warning(traceback.format_exc())
    
    def get_data_date_range(self, market: str = 'csi300') -> Dict[str, str]:
        """
        Get the available date range for the specified market.
        
        Args:
            market: Market name (e.g., 'csi300', 'csi500')
            
        Returns:
            Dictionary with 'start_date' and 'end_date' keys
        """
        self._init_qlib()
        
        if not self._qlib_initialized:
            return {'start_date': None, 'end_date': None, 'error': 'Qlib not initialized'}
        
        try:
            from qlib.data import D
            
            # Get a sample stock to check date range
            # Try to get calendar/trading days from Qlib
            instruments = D.instruments(market)
            
            # Get list of instruments
            instrument_list = D.list_instruments(instruments, as_list=True)
            if not instrument_list:
                return {'start_date': None, 'end_date': None, 'error': f'No instruments in {market}'}
            
            # Get features for a sample stock to find date range
            sample_stock = instrument_list[0]
            df = D.features([sample_stock], fields=['$close'], 
                           start_time='2000-01-01', end_time='2099-12-31')
            
            if df.empty:
                return {'start_date': None, 'end_date': None, 'error': 'No data available'}
            
            # Get date range from index
            dates = df.index.get_level_values('datetime')
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'num_instruments': len(instrument_list)
            }
        except Exception as e:
            logger.error(f"Failed to get data date range: {e}")
            return {'start_date': None, 'end_date': None, 'error': str(e)}
    
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
            result['traceback'] = traceback.format_exc()
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"Training failed for {model_name}: {e}")
            logger.error(traceback.format_exc())
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
            
            # Generate predictions using SignalRecord (saves to recorder artifacts)
            rec = SignalRecord(model, dataset, recorder=R.get_recorder())
            rec.generate()
            
            # Get metrics
            metrics = R.get_recorder().list_metrics()
            
            # Get predictions directly from model (SignalRecord saves to artifacts)
            predictions = model.predict(dataset, segment="test")
        
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
        
        # Define processors for data handling
        learn_processors = [
            {"class": "DropnaLabel"},
        ]
        infer_processors = [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "ZScoreNorm", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
        ]
        
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
                    "label": ([label_expr], ["LABEL0"]),
                    "learn_processors": learn_processors,
                    "infer_processors": infer_processors,
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
                    "label": ([label_expr], ["LABEL0"]),
                    "learn_processors": learn_processors,
                    "infer_processors": infer_processors,
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
        
        # Handle both old format (just model) and new format (dict with model and config)
        if isinstance(data, dict):
            return data.get('model'), data.get('config')
        else:
            # Old format: data is the model itself
            logger.warning(f"Model saved in old format without config: {model_path}")
            return data, None
    
    def predict_stocks(
        self,
        feature_set: FeatureSet,
        model_type: ModelType,
        horizon: PredictionHorizon,
        stock_codes: List[str],
        predict_date: str = None,
        training_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Run predictions for a list of stocks using a trained model.
        
        Args:
            feature_set: Feature set used for training
            model_type: Model type
            horizon: Prediction horizon
            stock_codes: List of stock codes to predict
            predict_date: Date to predict for (default: latest available)
            training_id: Specific training ID (optional)
            
        Returns:
            List of prediction dictionaries with stock_code and prediction
        """
        self._init_qlib()
        
        from qlib.data.dataset import DatasetH
        from qlib.utils import init_instance_by_config
        
        # Load the trained model
        model, train_config = self.load_trained_model(
            feature_set=feature_set,
            model_type=model_type,
            horizon=horizon,
            training_id=training_id,
        )
        
        if model is None:
            raise ValueError("Failed to load model")
        
        # Determine date range for prediction
        # We need historical data to compute features
        from datetime import datetime, timedelta
        if predict_date:
            end_date = predict_date
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Need enough history for feature computation (at least 60 days for Alpha360)
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
        
        # Get fit times from training config for consistent normalization
        if train_config:
            fit_start = train_config.train_start
            fit_end = train_config.train_end
        else:
            # Fallback if no config saved
            fit_start = "2017-01-01"
            fit_end = "2022-12-31"
        
        # Create handler config for the specific stocks
        handler_config = self._get_handler_config_for_stocks(
            feature_set=feature_set,
            horizon=horizon,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            fit_start=fit_start,
            fit_end=fit_end,
        )
        
        # Create dataset for prediction
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "test": (start_date, end_date),
                },
            },
        }
        
        dataset = init_instance_by_config(dataset_config)
        
        # Run prediction
        predictions_series = model.predict(dataset, segment="test")
        
        # Get the latest predictions for each stock
        if predictions_series is not None and len(predictions_series) > 0:
            # Group by stock and get the latest prediction
            latest_predictions = predictions_series.groupby(level='instrument').last()
            
            results = []
            for stock_code in stock_codes:
                if stock_code in latest_predictions.index:
                    pred_value = float(latest_predictions[stock_code])
                    results.append({
                        'stock_code': stock_code,
                        'prediction': pred_value,
                    })
                else:
                    results.append({
                        'stock_code': stock_code,
                        'prediction': None,
                        'error': 'No data available'
                    })
            
            # Sort by prediction (descending)
            results.sort(key=lambda x: x.get('prediction') or float('-inf'), reverse=True)
            return results
        else:
            return [{'stock_code': code, 'prediction': None, 'error': 'Prediction failed'} for code in stock_codes]
    
    def predict_stocks_with_model(
        self,
        model_filename: str,
        stock_codes: List[str],
        predict_date: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Run predictions for a list of stocks using a specific model file.
        
        Args:
            model_filename: The filename of the model (e.g., 'alpha158_lgbmodel_horizon1_20240101_120000.pkl')
            stock_codes: List of stock codes to predict
            predict_date: Date to predict for (default: latest available)
            
        Returns:
            List of prediction dictionaries with stock_code and prediction
        """
        self._init_qlib()
        
        from qlib.data.dataset import DatasetH
        from qlib.utils import init_instance_by_config
        
        # Load the model directly from filename
        models_dir = os.path.join(self.train_output_path, 'models')
        
        # Ensure filename ends with .pkl
        if not model_filename.endswith('.pkl'):
            model_filename = model_filename + '.pkl'
        
        model_path = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_filename}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old and new format
        if isinstance(data, dict):
            model = data.get('model')
            train_config = data.get('config')
        else:
            model = data
            train_config = None
        
        if model is None:
            raise ValueError(f"Failed to load model from {model_filename}")
        
        # Parse model info from filename to get feature_set and horizon
        # Format: feature_set_model_type_horizonN_timestamp.pkl
        name_parts = model_filename.replace('.pkl', '').split('_')
        
        # Determine feature_set
        if 'alpha158' in model_filename:
            feature_set = FeatureSet.ALPHA158
        elif 'alpha360' in model_filename:
            feature_set = FeatureSet.ALPHA360
        else:
            feature_set = FeatureSet.ALPHA158  # default
        
        # Determine horizon from filename
        horizon = PredictionHorizon.DAY_1  # default
        for part in name_parts:
            if part.startswith('horizon'):
                try:
                    h_val = int(part.replace('horizon', ''))
                    horizon = PredictionHorizon(h_val)
                except (ValueError, KeyError):
                    pass
        
        # Override with config values if available
        if train_config:
            feature_set = train_config.feature_set
            horizon = train_config.horizon
        
        logger.info(f"Loaded model {model_filename}: feature_set={feature_set.value}, horizon={horizon.value}")
        
        # Determine date range for prediction
        from datetime import datetime, timedelta
        if predict_date:
            end_date = predict_date
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Need enough history for feature computation
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
        
        # Get fit times from training config for consistent normalization
        if train_config:
            fit_start = train_config.train_start
            fit_end = train_config.train_end
        else:
            fit_start = "2017-01-01"
            fit_end = "2022-12-31"
        
        # Create handler config for the specific stocks
        handler_config = self._get_handler_config_for_stocks(
            feature_set=feature_set,
            horizon=horizon,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            fit_start=fit_start,
            fit_end=fit_end,
        )
        
        # Create dataset for prediction
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "test": (start_date, end_date),
                },
            },
        }
        
        dataset = init_instance_by_config(dataset_config)
        
        # Run prediction
        predictions_series = model.predict(dataset, segment="test")
        
        # Get the latest predictions for each stock
        if predictions_series is not None and len(predictions_series) > 0:
            latest_predictions = predictions_series.groupby(level='instrument').last()
            
            results = []
            for stock_code in stock_codes:
                if stock_code in latest_predictions.index:
                    pred_value = float(latest_predictions[stock_code])
                    results.append({
                        'stock_code': stock_code,
                        'prediction': pred_value,
                    })
                else:
                    results.append({
                        'stock_code': stock_code,
                        'prediction': None,
                        'error': 'No data available'
                    })
            
            results.sort(key=lambda x: x.get('prediction') or float('-inf'), reverse=True)
            return results
        else:
            return [{'stock_code': code, 'prediction': None, 'error': 'Prediction failed'} for code in stock_codes]
    
    def _get_handler_config_for_stocks(
        self,
        feature_set: FeatureSet,
        horizon: PredictionHorizon,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        fit_start: str = None,
        fit_end: str = None,
    ) -> Dict:
        """Get handler configuration for specific stocks.
        
        Args:
            feature_set: Feature set type
            horizon: Prediction horizon
            stock_codes: List of stock codes
            start_date: Data start date
            end_date: Data end date
            fit_start: Fit start date for normalization (should match training)
            fit_end: Fit end date for normalization (should match training)
        """
        label_expression = f"Ref($close, -{horizon.value})/Ref($close, -1) - 1"
        
        # Use training fit period if provided, otherwise use data range
        fit_start_time = fit_start or start_date
        fit_end_time = fit_end or end_date
        
        # Define processors for data handling
        learn_processors = [
            {"class": "DropnaLabel"},
        ]
        infer_processors = [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "ZScoreNorm", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
        ]
        
        if feature_set == FeatureSet.ALPHA158:
            handler_class = "Alpha158"
        else:
            handler_class = "Alpha360"
        
        return {
            "class": handler_class,
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "instruments": stock_codes,
                "start_time": start_date,
                "end_time": end_date,
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
                "label": ([label_expression], ["LABEL0"]),
                "learn_processors": learn_processors,
                "infer_processors": infer_processors,
            },
        }
    
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
        
        # Check data availability before proceeding
        market = kwargs.get('market', original_config.market if original_config else 'csi300')
        data_range = self.get_data_date_range(market)
        if data_range.get('start_date') and data_range.get('end_date'):
            data_start = data_range['start_date']
            data_end = data_range['end_date']
            logger.info(f"Available data range for {market}: {data_start} to {data_end}")
            
            # Calculate extended end date needed for label calculation
            from datetime import datetime as dt, timedelta
            end_date_dt = dt.strptime(finetune_end, "%Y-%m-%d")
            extended_end_dt = end_date_dt + timedelta(days=horizon.value + 4)
            
            if finetune_start < data_start or extended_end_dt.strftime('%Y-%m-%d') > data_end:
                return {
                    'status': 'failed',
                    'error': f'Date range out of available data. Available: {data_start} to {data_end}. '
                            f'Requested: {finetune_start} to {finetune_end} (needs data until '
                            f'{extended_end_dt.strftime("%Y-%m-%d")} for label calculation).',
                    'message': 'Please adjust your date range or download more recent data.'
                }
        
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
            result['traceback'] = traceback.format_exc()
            result['end_time'] = datetime.now().isoformat()
            logger.error(f"Finetuning failed for {model_name}: {e}")
            logger.error(traceback.format_exc())
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
        
        model_name = f"{feature_set.value}_{model_type.value}_horizon{horizon.value}"
        market = kwargs.get('market', original_config.market if original_config else 'csi300')
        
        # Get the label expression for this horizon
        label_expr = f"Ref($close, -{horizon.value})/Ref($close, -1) - 1"
        
        # Create a simplified handler config for finetuning
        if feature_set == FeatureSet.ALPHA158:
            handler_class = "Alpha158"
        else:
            handler_class = "Alpha360"
        
        # Need to extend end_time to accommodate label calculation
        # Label uses Ref($close, -N) which needs N days of future data
        from datetime import datetime as dt, timedelta
        end_date_dt = dt.strptime(finetune_end, "%Y-%m-%d")
        extended_end = (end_date_dt + timedelta(days=horizon.value + 10)).strftime("%Y-%m-%d")
        
        handler_config = {
            "class": handler_class,
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": finetune_start,
                "end_time": extended_end,  # Extended to have data for label calculation
                "fit_start_time": finetune_start,
                "fit_end_time": finetune_end,
                "instruments": market,
                "label": ([label_expr], ["LABEL0"]),
                "learn_processors": [
                    {"class": "DropnaLabel"},
                ],
                "infer_processors": [
                    {"class": "ProcessInf", "kwargs": {}},
                    {"class": "ZScoreNorm", "kwargs": {}},
                    {"class": "Fillna", "kwargs": {}},
                ],
            },
        }
        
        # Create dataset for finetuning - LGBModel.finetune requires both train and valid segments
        # Split the finetune period: 80% train, 20% valid
        from datetime import datetime as dt, timedelta
        start_dt = dt.strptime(finetune_start, "%Y-%m-%d")
        end_dt = dt.strptime(finetune_end, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        train_days = int(total_days * 0.8)
        split_date = (start_dt + timedelta(days=train_days)).strftime("%Y-%m-%d")
        valid_start_date = (start_dt + timedelta(days=train_days + 1)).strftime("%Y-%m-%d")
        
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": (finetune_start, split_date),
                    "valid": (valid_start_date, finetune_end),
                },
            },
        }
        
        logger.info(f"Creating finetune dataset: train={finetune_start} to {split_date}, valid={valid_start_date} to {finetune_end}, extended_end={extended_end}")
        dataset = init_instance_by_config(dataset_config)
        
        # Check if dataset has data
        try:
            train_data = dataset.prepare("train", col_set="feature")
            if train_data is None or (hasattr(train_data, 'empty') and train_data.empty) or len(train_data) == 0:
                raise ValueError(f"No data available for finetuning period {finetune_start} to {finetune_end}. "
                               f"Please check if your data covers this date range.")
            logger.info(f"Finetune dataset has {len(train_data)} samples")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare finetune dataset: {e}")
            raise ValueError(f"Failed to prepare finetune dataset: {e}. "
                           f"Check if data exists for {finetune_start} to {finetune_end}")
        
        # Finetuning
        experiment_name = f"{model_name}_finetune_{finetune_id}"
        with R.start(experiment_name=experiment_name):
            # For LGBModel, the finetune() method has bugs in some Qlib versions
            # So we retrain with the new data instead
            if model_type == ModelType.LGBMODEL:
                from qlib.contrib.model.gbdt import LGBModel as QlibLGBModel
                # Create a new model and train on finetune data
                # This effectively "finetunes" by training a fresh model on recent data
                model = QlibLGBModel(
                    loss="mse",
                    colsample_bytree=0.8879,
                    learning_rate=0.0421,
                    subsample=0.8789,
                    lambda_l1=205.6999,
                    lambda_l2=580.9768,
                    max_depth=8,
                    num_leaves=210,
                    num_threads=20,
                    early_stopping_rounds=50,
                    num_boost_round=200,  # Fewer rounds for finetuning
                )
                model.fit(dataset)
            else:
                # Transformer - continue training with fewer epochs
                model.fit(dataset)
            
            metrics = R.get_recorder().list_metrics()
        
        # Save finetuned model
        model_save_path = os.path.join(
            self.train_output_path, 'models',
            f"{model_name}_finetuned_{finetune_id}.pkl"
        )
        
        # Create a config object for the finetuned model
        finetuned_config = TrainingConfig(
            feature_set=feature_set,
            model_type=model_type,
            horizon=horizon,
            market=market,
            train_start=finetune_start,
            train_end=finetune_end,
        )
        
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'config': finetuned_config,
                'original_model': model_name,
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
                        'name': filename.replace('.pkl', ''),
                        'path': filepath,
                        'created': datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).isoformat(),
                        'timestamp': datetime.fromtimestamp(
                            os.path.getctime(filepath)
                        ).strftime('%Y-%m-%d %H:%M'),
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
    
    def delete_model(self, filename: str) -> Dict[str, Any]:
        """
        Delete a saved model file.
        
        Args:
            filename: The filename of the model to delete (e.g., 'alpha158_lgbmodel_horizon1_20240101_120000.pkl')
            
        Returns:
            Dictionary with status and message
        """
        models_dir = os.path.join(self.train_output_path, 'models')
        
        # Ensure filename ends with .pkl
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        
        filepath = os.path.join(models_dir, filename)
        
        # Security check: ensure the path is within models_dir
        if not os.path.abspath(filepath).startswith(os.path.abspath(models_dir)):
            return {
                'status': 'error',
                'message': 'Invalid filename'
            }
        
        if not os.path.exists(filepath):
            return {
                'status': 'error',
                'message': f'Model not found: {filename}'
            }
        
        try:
            os.remove(filepath)
            logger.info(f"Deleted model: {filepath}")
            
            # Also try to delete associated prediction file if exists
            pred_filename = filename.replace('.pkl', '_predictions.pkl')
            pred_filepath = os.path.join(self.train_output_path, 'predictions', pred_filename)
            if os.path.exists(pred_filepath):
                os.remove(pred_filepath)
                logger.info(f"Deleted predictions: {pred_filepath}")
            
            return {
                'status': 'success',
                'message': f'Model {filename} deleted successfully'
            }
        except Exception as e:
            logger.error(f"Failed to delete model {filename}: {e}")
            return {
                'status': 'error',
                'message': f'Failed to delete model: {str(e)}'
            }
    
    def get_finetuned_models(self) -> List[Dict[str, Any]]:
        """List all finetuned models."""
        finetuned = []
        models_dir = os.path.join(self.train_output_path, 'models')
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl') and 'finetuned' in filename:
                    filepath = os.path.join(models_dir, filename)
                    
                    # Try to load model info
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        
                        finetuned.append({
                            'name': filename.replace('.pkl', ''),
                            'original_model': data.get('original_model', 'Unknown'),
                            'finetune_start': data.get('finetune_start', 'N/A'),
                            'finetune_end': data.get('finetune_end', 'N/A'),
                            'timestamp': data.get('timestamp', 'N/A'),
                            'status': 'completed',
                        })
                    except Exception:
                        finetuned.append({
                            'name': filename.replace('.pkl', ''),
                            'original_model': 'Unknown',
                            'finetune_start': 'N/A',
                            'finetune_end': 'N/A',
                            'timestamp': datetime.fromtimestamp(
                                os.path.getctime(filepath)
                            ).isoformat(),
                            'status': 'completed',
                        })
        
        return sorted(finetuned, key=lambda x: x['timestamp'], reverse=True)
    
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
