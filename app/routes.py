"""
Flask routes for the web application
"""
from flask import Blueprint, render_template, jsonify, request
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib_module.trainer import (
    QlibTrainer, 
    FeatureSet, 
    ModelType, 
    PredictionHorizon,
    TrainingConfig
)
from qlib_module.data_manager import DataManager

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@main_bp.route('/dashboard')
def dashboard():
    """Dashboard page showing training status and results."""
    data_manager = DataManager()
    trainer = QlibTrainer()
    
    context = {
        'data_status': data_manager.get_data_status(),
        'training_history': trainer.get_training_history(),
        'last_update': data_manager.get_last_update_time(),
        'models': trainer.list_models(),
        'feature_sets': [fs.value for fs in FeatureSet],
        'model_types': [mt.value for mt in ModelType],
        'horizons': [h.value for h in PredictionHorizon],
    }
    return render_template('dashboard.html', **context)


@main_bp.route('/predict')
def predict_page():
    """Stock prediction page."""
    from datetime import date
    trainer = QlibTrainer()
    
    context = {
        'models': trainer.list_models(),
        'feature_sets': [fs.value for fs in FeatureSet],
        'model_types': [mt.value for mt in ModelType],
        'horizons': [h.value for h in PredictionHorizon],
        'today': date.today().isoformat(),
    }
    return render_template('predict.html', **context)


@main_bp.route('/api/predict', methods=['POST'])
def run_prediction():
    """API endpoint to run predictions for a list of stocks."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        # Get parameters
        feature_set = FeatureSet(data.get('feature_set', 'alpha158'))
        model_type = ModelType(data.get('model_type', 'lgbmodel'))
        horizon = PredictionHorizon(int(data.get('horizon', 1)))
        predict_date = data.get('predict_date')
        stock_codes = data.get('stock_codes', [])
        
        if not stock_codes:
            return jsonify({
                'status': 'error',
                'message': 'No stock codes provided'
            }), 400
        
        # Run prediction
        predictions = trainer.predict_stocks(
            feature_set=feature_set,
            model_type=model_type,
            horizon=horizon,
            stock_codes=stock_codes,
            predict_date=predict_date,
        )
        
        model_name = f"{feature_set.value}_{model_type.value}_horizon{horizon.value}"
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'predictions': predictions,
            'count': len(predictions)
        })
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'message': f'Model not found: {str(e)}'
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/train', methods=['POST'])
def start_training():
    """API endpoint to start model training for a specific configuration."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        # Get parameters from request
        feature_set = data.get('feature_set', 'alpha158')
        model_type = data.get('model_type', 'lgbmodel')
        horizon = data.get('horizon', 1)
        
        # Create config
        config = TrainingConfig(
            feature_set=FeatureSet(feature_set),
            model_type=ModelType(model_type),
            horizon=PredictionHorizon(int(horizon)),
            market=data.get('market', 'csi300'),
        )
        
        result = trainer.train_model(config)
        return jsonify({
            'status': 'success',
            'message': f'Training started for {config.get_model_name()}',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/train/all', methods=['POST'])
def train_all_models():
    """API endpoint to train all model combinations."""
    try:
        trainer = QlibTrainer()
        result = trainer.train_all_models()
        return jsonify({
            'status': 'success',
            'message': 'Training all models started',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/finetune', methods=['POST'])
def finetune_model():
    """API endpoint to finetune a trained model with newest data."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        # Get parameters from request
        feature_set = FeatureSet(data.get('feature_set', 'alpha158'))
        model_type = ModelType(data.get('model_type', 'lgbmodel'))
        horizon = PredictionHorizon(int(data.get('horizon', 1)))
        finetune_start = data.get('finetune_start')
        finetune_end = data.get('finetune_end')
        
        result = trainer.finetune_model(
            feature_set=feature_set,
            model_type=model_type,
            horizon=horizon,
            finetune_start=finetune_start,
            finetune_end=finetune_end,
        )
        return jsonify({
            'status': 'success',
            'message': f'Finetuning started for {feature_set.value}_{model_type.value}_horizon{horizon.value}',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/finetune/all', methods=['POST'])
def finetune_all_models():
    """API endpoint to finetune all trained models with newest data."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        result = trainer.finetune_all_models(
            finetune_start=data.get('finetune_start'),
            finetune_end=data.get('finetune_end'),
        )
        return jsonify({
            'status': 'success',
            'message': 'Finetuning all models started',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/data/status')
def data_status():
    """API endpoint to get data status."""
    try:
        data_manager = DataManager()
        status = data_manager.get_data_status()
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/data/download', methods=['POST'])
def download_data():
    """API endpoint to manually trigger data download."""
    try:
        data_manager = DataManager()
        result = data_manager.download_data()
        return jsonify({
            'status': 'success',
            'message': 'Data download initiated',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/models')
def list_models():
    """API endpoint to list trained models."""
    try:
        trainer = QlibTrainer()
        models = trainer.list_models()
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/models/load', methods=['POST'])
def load_model():
    """API endpoint to load a trained model."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        feature_set = FeatureSet(data.get('feature_set', 'alpha158'))
        model_type = ModelType(data.get('model_type', 'lgbmodel'))
        horizon = PredictionHorizon(int(data.get('horizon', 1)))
        training_id = data.get('training_id')
        
        model, config = trainer.load_trained_model(
            feature_set=feature_set,
            model_type=model_type,
            horizon=horizon,
            training_id=training_id,
        )
        
        return jsonify({
            'status': 'success',
            'message': f'Model loaded successfully',
            'model_name': f"{feature_set.value}_{model_type.value}_horizon{horizon.value}",
            'has_model': model is not None,
        })
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/training/history')
def training_history():
    """API endpoint to get training history."""
    try:
        trainer = QlibTrainer()
        history = trainer.get_training_history()
        return jsonify({
            'status': 'success',
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/training/combinations')
def training_combinations():
    """API endpoint to get available training combinations."""
    return jsonify({
        'status': 'success',
        'feature_sets': [fs.value for fs in FeatureSet],
        'model_types': [mt.value for mt in ModelType],
        'horizons': [{'value': h.value, 'name': h.name} for h in PredictionHorizon],
        'combinations': [
            {
                'feature_set': fs.value,
                'model_type': mt.value,
                'horizons': [h.value for h in PredictionHorizon]
            }
            for fs, mt in QlibTrainer.TRAINING_COMBINATIONS
        ]
    })
