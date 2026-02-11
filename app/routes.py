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


@main_bp.route('/data')
def data_page():
    """Data management page."""
    data_manager = DataManager()
    
    context = {
        'data_status': data_manager.get_data_status(),
        'last_update': data_manager.get_last_update_time(),
        'validation': data_manager.validate_data(),
        'download_history': data_manager.get_download_history(),
    }
    return render_template('data.html', **context)


@main_bp.route('/predict')
def predict_page():
    """Stock prediction page."""
    from datetime import date
    trainer = QlibTrainer()
    
    # Get all models including finetuned ones
    models = trainer.list_models()
    
    context = {
        'models': models,
        'today': date.today().isoformat(),
    }
    return render_template('predict.html', **context)


@main_bp.route('/train')
def train_page():
    """Model training page."""
    trainer = QlibTrainer()
    
    context = {
        'models': trainer.list_models(),
        'feature_sets': [fs.value for fs in FeatureSet],
        'model_types': [mt.value for mt in ModelType],
        'horizons': [{'value': h.value, 'name': h.name} for h in PredictionHorizon],
    }
    return render_template('train.html', **context)


@main_bp.route('/finetune')
def finetune_page():
    """Model finetuning page."""
    from datetime import date, timedelta
    trainer = QlibTrainer()
    
    # Default finetune period: last 3 months
    today = date.today()
    finetune_end = today.isoformat()
    finetune_start = (today - timedelta(days=90)).isoformat()
    
    # Get models with parsed info
    models = trainer.list_models()
    for model in models:
        # Parse model name to extract feature_set, model_type, horizon
        # Format: feature_set_model_type_horizonN
        parts = model.get('name', '').split('_')
        if len(parts) >= 3:
            model['feature_set'] = parts[0]
            model['model_type'] = parts[1]
            # Extract horizon number from 'horizonN'
            horizon_part = parts[2] if len(parts) > 2 else 'horizon1'
            model['horizon'] = int(horizon_part.replace('horizon', '')) if 'horizon' in horizon_part else 1
    
    # Get finetuned models history
    finetuned_models = trainer.get_finetuned_models() if hasattr(trainer, 'get_finetuned_models') else []
    
    context = {
        'models': models,
        'finetuned_models': finetuned_models,
        'finetune_start': finetune_start,
        'finetune_end': finetune_end,
    }
    return render_template('finetune.html', **context)


@main_bp.route('/api/predict', methods=['POST'])
def run_prediction():
    """API endpoint to run predictions for a list of stocks."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        # Get parameters
        model_filename = data.get('model_filename')  # New: use model filename directly
        predict_date = data.get('predict_date')
        stock_codes = data.get('stock_codes', [])
        
        if not stock_codes:
            return jsonify({
                'status': 'error',
                'message': 'No stock codes provided'
            }), 400
        
        if not model_filename:
            return jsonify({
                'status': 'error',
                'message': 'No model selected'
            }), 400
        
        # Run prediction using model filename
        predictions = trainer.predict_stocks_with_model(
            model_filename=model_filename,
            stock_codes=stock_codes,
            predict_date=predict_date,
        )
        
        model_name = model_filename.replace('.pkl', '')
        
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
        
        # Create config with optional date parameters
        config = TrainingConfig(
            feature_set=FeatureSet(feature_set),
            model_type=ModelType(model_type),
            horizon=PredictionHorizon(int(horizon)),
            market=data.get('market', 'csi300'),
            train_start=data.get('train_start', '2017-01-01'),
            train_end=data.get('train_end', '2022-12-31'),
            valid_start=data.get('valid_start', '2023-01-01'),
            valid_end=data.get('valid_end', '2023-06-30'),
            test_start=data.get('test_start', '2023-07-01'),
            test_end=data.get('test_end', '2024-12-31'),
            # Transformer-specific parameters
            num_layers=int(data.get('num_layers', 8)),
            early_stop=int(data.get('early_stop', 30)),
            batch_size=int(data.get('batch_size', 4096)),
            d_model=int(data.get('d_model', 256)),
            learning_rate=float(data.get('learning_rate', 0.0001)),
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
    import threading
    
    def run_download():
        try:
            data_manager = DataManager()
            data_manager.download_data()
        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.error(f"Download failed: {e}")
    
    # Check if download is already in progress
    progress = DataManager.get_download_progress()
    if progress['status'] == 'downloading':
        return jsonify({
            'status': 'error',
            'message': 'Download already in progress'
        }), 400
    
    # Start download in background thread
    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Download started in background'
    })


@main_bp.route('/api/data/download/progress')
def download_progress():
    """API endpoint to get current download progress."""
    try:
        progress = DataManager.get_download_progress()
        return jsonify({
            'status': 'success',
            'progress': progress
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/data/daterange')
def data_date_range():
    """API endpoint to get available data date range."""
    try:
        trainer = QlibTrainer()
        market = request.args.get('market', 'csi300')
        date_range = trainer.get_data_date_range(market)
        return jsonify({
            'status': 'success',
            'data': date_range
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/data/history')
def data_history():
    """API endpoint to get download history."""
    try:
        data_manager = DataManager()
        history = data_manager.get_download_history()
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


@main_bp.route('/api/models/delete', methods=['POST'])
def delete_model():
    """API endpoint to delete a trained model."""
    try:
        trainer = QlibTrainer()
        data = request.get_json(force=True, silent=True) or {}
        
        filename = data.get('filename')
        if not filename:
            return jsonify({
                'status': 'error',
                'message': 'filename is required'
            }), 400
        
        result = trainer.delete_model(filename)
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/models/detail')
def get_model_detail():
    """API endpoint to get detailed information about a trained model."""
    try:
        trainer = QlibTrainer()
        filename = request.args.get('filename')
        
        if not filename:
            return jsonify({
                'status': 'error',
                'message': 'filename is required'
            }), 400
        
        detail = trainer.get_model_detail(filename)
        
        if detail:
            return jsonify({
                'status': 'success',
                'model': detail
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model not found'
            }), 404
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


# =====================
# Stock Crawler API
# =====================

@main_bp.route('/api/instruments/list')
def list_instruments():
    """API endpoint to list available instrument files."""
    try:
        data_manager = DataManager()
        instruments_path = os.path.join(data_manager.data_path, 'instruments')
        
        if not os.path.exists(instruments_path):
            return jsonify({
                'status': 'error',
                'message': 'Instruments folder not found'
            }), 404
        
        instruments = []
        for entry in os.scandir(instruments_path):
            if entry.is_file() and entry.name.endswith('.txt'):
                name = entry.name.replace('.txt', '')
                # Count lines (stocks) in file
                try:
                    with open(entry.path, 'r') as f:
                        line_count = sum(1 for line in f if line.strip())
                except:
                    line_count = 0
                
                instruments.append({
                    'name': name,
                    'filename': entry.name,
                    'stocks_count': line_count
                })
        
        # Sort by name
        instruments.sort(key=lambda x: x['name'])
        
        return jsonify({
            'status': 'success',
            'instruments': instruments
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/instruments/stocks')
def get_instrument_stocks():
    """API endpoint to get stock codes from an instrument file for a given date range."""
    try:
        from datetime import datetime
        
        instrument = request.args.get('instrument', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        if not instrument:
            return jsonify({
                'status': 'error',
                'message': 'Instrument name is required'
            }), 400
        
        data_manager = DataManager()
        instrument_file = os.path.join(data_manager.data_path, 'instruments', f'{instrument}.txt')
        
        if not os.path.exists(instrument_file):
            return jsonify({
                'status': 'error',
                'message': f'Instrument file not found: {instrument}'
            }), 404
        
        # Parse the instrument file
        # Format: STOCK_CODE\tSTART_DATE\tEND_DATE
        stocks = set()  # Use set to avoid duplicates
        
        # Parse user-provided dates
        if start_date:
            try:
                query_start = datetime.strptime(start_date, '%Y-%m-%d')
            except:
                query_start = None
        else:
            query_start = None
            
        if end_date:
            try:
                query_end = datetime.strptime(end_date, '%Y-%m-%d')
            except:
                query_end = None
        else:
            query_end = None
        
        with open(instrument_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    stock_code = parts[0]
                    stock_start = parts[1]
                    stock_end = parts[2]
                    
                    # Filter by date range if provided
                    # Stock is valid if its active period overlaps with query period
                    if query_start or query_end:
                        try:
                            s_start = datetime.strptime(stock_start, '%Y-%m-%d')
                            s_end = datetime.strptime(stock_end, '%Y-%m-%d')
                            
                            # Check overlap
                            if query_end and s_start > query_end:
                                continue  # Stock starts after query end
                            if query_start and s_end < query_start:
                                continue  # Stock ends before query start
                        except:
                            pass  # Include if date parsing fails
                    
                    # Convert stock code format: SZ000001 -> sz000001
                    stock_code = stock_code.lower()
                    stocks.add(stock_code)
        
        stocks_list = sorted(list(stocks))
        
        return jsonify({
            'status': 'success',
            'instrument': instrument,
            'stocks': stocks_list,
            'count': len(stocks_list)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/crawler/fetch', methods=['POST'])
def crawler_fetch():
    """API endpoint to crawl stock data in real-time."""
    try:
        from qlib_module.stock_crawler import StockCrawler
        
        data = request.get_json(force=True, silent=True) or {}
        
        # Get parameters
        stock_codes = data.get('stock_codes', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not stock_codes:
            return jsonify({
                'status': 'error',
                'message': 'No stock codes provided'
            }), 400
        
        # Validate stock codes format - convert string to list if needed
        if isinstance(stock_codes, str):
            # Split by comma, semicolon, space, or newline
            import re
            stock_codes = [code.strip() for code in re.split(r'[,;\s\n]+', stock_codes) if code.strip()]
        
        if not stock_codes:
            return jsonify({
                'status': 'error',
                'message': 'Invalid stock codes format'
            }), 400
        
        # Check source - instrument source allows more stocks
        source = data.get('source', 'manual')
        max_stocks = 500 if source == 'instrument' else 10
        
        if len(stock_codes) > max_stocks:
            return jsonify({
                'status': 'error',
                'message': f'Too many stock codes. Maximum {max_stocks} per request.'
            }), 400
        
        # Reduce delay for large batches
        delay = 0.1 if len(stock_codes) > 20 else 0.3
        crawler = StockCrawler(delay=delay)
        results = {}
        errors = []
        
        for code in stock_codes:
            try:
                df = crawler.fetch_stock_data(code, start_date, end_date)
                if df is not None and not df.empty:
                    formatted_code = crawler.format_stock_code(code, format_type='qlib')
                    # Convert DataFrame to JSON-serializable format
                    df_dict = df.copy()
                    df_dict['date'] = df_dict['date'].dt.strftime('%Y-%m-%d')
                    results[formatted_code] = {
                        'records': df_dict.to_dict(orient='records'),
                        'count': len(df_dict),
                        'date_range': {
                            'start': df_dict['date'].min(),
                            'end': df_dict['date'].max()
                        }
                    }
                else:
                    errors.append({'code': code, 'error': 'No data available'})
            except Exception as e:
                errors.append({'code': code, 'error': str(e)})
        
        return jsonify({
            'status': 'success' if results else 'error',
            'data': results,
            'errors': errors,
            'success_count': len(results),
            'error_count': len(errors)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/crawler/fetch-single', methods=['POST'])
def crawler_fetch_single():
    """API endpoint to crawl single stock data with full details."""
    try:
        from qlib_module.stock_crawler import StockCrawler
        
        data = request.get_json(force=True, silent=True) or {}
        
        stock_code = data.get('stock_code', '').strip()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not stock_code:
            return jsonify({
                'status': 'error',
                'message': 'Stock code is required'
            }), 400
        
        crawler = StockCrawler(delay=0.1)
        df = crawler.fetch_stock_data(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            return jsonify({
                'status': 'error',
                'message': f'No data available for {stock_code}'
            }), 404
        
        # Convert DataFrame to JSON-serializable format
        df_dict = df.copy()
        df_dict['date'] = df_dict['date'].dt.strftime('%Y-%m-%d')
        
        formatted_code = crawler.format_stock_code(stock_code, format_type='qlib')
        
        return jsonify({
            'status': 'success',
            'stock_code': formatted_code,
            'records': df_dict.to_dict(orient='records'),
            'count': len(df_dict),
            'date_range': {
                'start': df_dict['date'].min(),
                'end': df_dict['date'].max()
            },
            'columns': list(df_dict.columns)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/crawler/validate-code', methods=['POST'])
def crawler_validate_code():
    """API endpoint to validate and normalize stock code."""
    try:
        from qlib_module.stock_crawler import StockCrawler
        
        data = request.get_json(force=True, silent=True) or {}
        stock_code = data.get('stock_code', '').strip()
        
        if not stock_code:
            return jsonify({
                'status': 'error',
                'message': 'Stock code is required'
            }), 400
        
        market_code, pure_code = StockCrawler.normalize_stock_code(stock_code)
        qlib_format = StockCrawler.format_stock_code(stock_code, format_type='qlib')
        suffix_format = StockCrawler.format_stock_code(stock_code, format_type='suffix')
        
        # Determine market name
        market_names = {1: 'Shanghai (SSE)', 0: 'Shenzhen (SZSE)'}
        if pure_code.startswith(('4', '8')) and len(pure_code) == 6:
            market_name = 'Beijing (BSE)'
        else:
            market_name = market_names.get(market_code, 'Unknown')
        
        return jsonify({
            'status': 'success',
            'original': stock_code,
            'normalized': {
                'market_code': market_code,
                'pure_code': pure_code,
                'qlib_format': qlib_format,
                'suffix_format': suffix_format,
                'market_name': market_name
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/crawler/merge', methods=['POST'])
def crawler_merge_to_qlib():
    """API endpoint to merge crawled data into existing qlib binary files."""
    try:
        from qlib_module.stock_crawler import StockCrawler, QlibDataMerger
        import pandas as pd
        
        data = request.get_json(force=True, silent=True) or {}
        crawled_data = data.get('crawled_data', {})
        
        if not crawled_data:
            return jsonify({
                'status': 'error',
                'message': 'No crawled data provided'
            }), 400
        
        # Get qlib data path
        data_manager = DataManager()
        qlib_data_path = data_manager.data_path
        
        # Convert the crawled data back to DataFrames
        data_dict = {}
        for stock_code, stock_data in crawled_data.items():
            records = stock_data.get('records', [])
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                data_dict[stock_code] = df
        
        if not data_dict:
            return jsonify({
                'status': 'error',
                'message': 'No valid stock data to merge'
            }), 400
        
        # Merge into qlib format
        merger = QlibDataMerger(qlib_data_path)
        result = merger.merge_multiple_stocks(data_dict)
        
        return jsonify({
            'status': 'success',
            'message': f"Merged {result['success_count']} stocks into qlib data",
            'result': {
                'success_count': result['success_count'],
                'failed_count': result['failed_count'],
                'total_records_added': result['total_records_added'],
                'total_records_updated': result['total_records_updated'],
                'total_dates_added': result.get('total_dates_added', 0),
                'status': result['status']
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# =====================
# Data Browser API
# =====================

@main_bp.route('/api/data/browse')
def browse_data_folder():
    """API endpoint to browse folders and files in cn_data directory with pagination."""
    try:
        import os
        from pathlib import Path
        
        data_manager = DataManager()
        base_path = Path(data_manager.data_path)
        
        # Get relative path from query parameter
        rel_path = request.args.get('path', '')
        
        # Pagination parameters
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 10))
        
        # Security: prevent path traversal
        if '..' in rel_path:
            return jsonify({
                'status': 'error',
                'message': 'Invalid path'
            }), 400
        
        current_path = base_path / rel_path if rel_path else base_path
        
        if not current_path.exists():
            return jsonify({
                'status': 'error',
                'message': 'Path does not exist'
            }), 404
        
        if not str(current_path.resolve()).startswith(str(base_path.resolve())):
            return jsonify({
                'status': 'error',
                'message': 'Access denied'
            }), 403
        
        all_items = []
        
        if current_path.is_dir():
            # Get all items sorted by name (ascending) - use scandir for speed
            with os.scandir(current_path) as entries:
                sorted_entries = sorted(entries, key=lambda x: x.name.lower())
                
                for entry in sorted_entries:
                    item_info = {
                        'name': entry.name,
                        'type': 'folder' if entry.is_dir() else 'file',
                        'path': str(Path(entry.path).relative_to(base_path)).replace('\\', '/'),
                    }
                    
                    if entry.is_file():
                        item_info['size'] = entry.stat().st_size
                        item_info['extension'] = Path(entry.name).suffix
                    
                    all_items.append(item_info)
        
        # Apply pagination
        total_items = len(all_items)
        paginated_items = all_items[offset:offset + limit]
        has_more = (offset + limit) < total_items
        
        # Build breadcrumb
        breadcrumb = []
        if rel_path:
            parts = rel_path.split('/')
            for i, part in enumerate(parts):
                breadcrumb.append({
                    'name': part,
                    'path': '/'.join(parts[:i+1])
                })
        
        return jsonify({
            'status': 'success',
            'current_path': rel_path or '/',
            'breadcrumb': breadcrumb,
            'items': paginated_items,
            'is_root': not rel_path,
            'total_items': total_items,
            'offset': offset,
            'limit': limit,
            'has_more': has_more
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/data/read-bin')
def read_bin_file():
    """API endpoint to read and decode a qlib binary file."""
    try:
        import numpy as np
        from pathlib import Path
        
        data_manager = DataManager()
        base_path = Path(data_manager.data_path)
        
        # Get file path from query parameter
        file_path = request.args.get('path', '')
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 100))
        
        # Security: prevent path traversal
        if '..' in file_path:
            return jsonify({
                'status': 'error',
                'message': 'Invalid path'
            }), 400
        
        full_path = base_path / file_path
        
        if not full_path.exists():
            return jsonify({
                'status': 'error',
                'message': 'File does not exist'
            }), 404
        
        if not str(full_path.resolve()).startswith(str(base_path.resolve())):
            return jsonify({
                'status': 'error',
                'message': 'Access denied'
            }), 403
        
        if not full_path.suffix == '.bin':
            return jsonify({
                'status': 'error',
                'message': 'Only .bin files can be read'
            }), 400
        
        # Read binary data as float32 (qlib format)
        data = np.fromfile(str(full_path), dtype='float32')
        total_count = len(data)
        
        # Get calendar to map indices to dates
        calendar_file = base_path / 'calendars' / 'day.txt'
        calendar = []
        if calendar_file.exists():
            with open(calendar_file, 'r') as f:
                calendar = [line.strip() for line in f if line.strip()]
        
        # Get slice of data
        end_idx = min(offset + limit, total_count)
        data_slice = data[offset:end_idx]
        
        # Build records with date mapping
        records = []
        for i, value in enumerate(data_slice):
            idx = offset + i
            record = {
                'index': idx,
                'value': None if np.isnan(value) else round(float(value), 4),
                'date': calendar[idx] if idx < len(calendar) else None
            }
            records.append(record)
        
        # Calculate statistics
        valid_data = data[~np.isnan(data)]
        stats = {
            'total_count': total_count,
            'valid_count': len(valid_data),
            'nan_count': total_count - len(valid_data),
            'min': round(float(np.min(valid_data)), 4) if len(valid_data) > 0 else None,
            'max': round(float(np.max(valid_data)), 4) if len(valid_data) > 0 else None,
            'mean': round(float(np.mean(valid_data)), 4) if len(valid_data) > 0 else None,
        }
        
        # Find first and last valid data indices
        valid_indices = np.where(~np.isnan(data))[0]
        if len(valid_indices) > 0:
            stats['first_valid_index'] = int(valid_indices[0])
            stats['last_valid_index'] = int(valid_indices[-1])
            stats['first_valid_date'] = calendar[valid_indices[0]] if valid_indices[0] < len(calendar) else None
            stats['last_valid_date'] = calendar[valid_indices[-1]] if valid_indices[-1] < len(calendar) else None
        
        return jsonify({
            'status': 'success',
            'file_path': file_path,
            'file_size': full_path.stat().st_size,
            'offset': offset,
            'limit': limit,
            'records': records,
            'stats': stats,
            'has_more': end_idx < total_count
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@main_bp.route('/api/data/calendar')
def get_calendar():
    """API endpoint to get the trading calendar."""
    try:
        from pathlib import Path
        
        data_manager = DataManager()
        base_path = Path(data_manager.data_path)
        
        calendar_type = request.args.get('type', 'day')
        calendar_file = base_path / 'calendars' / f'{calendar_type}.txt'
        
        if not calendar_file.exists():
            return jsonify({
                'status': 'error',
                'message': 'Calendar file not found'
            }), 404
        
        with open(calendar_file, 'r') as f:
            dates = [line.strip() for line in f if line.strip()]
        
        return jsonify({
            'status': 'success',
            'calendar_type': calendar_type,
            'total_dates': len(dates),
            'first_date': dates[0] if dates else None,
            'last_date': dates[-1] if dates else None,
            'dates': dates
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
