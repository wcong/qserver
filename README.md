# QServer - Qlib Training Server

A Python web server for quantitative trading model training using Microsoft's Qlib framework.

## Features

- 🌐 **Web Dashboard**: Flask-based web interface for monitoring and control
- 📊 **Qlib Integration**: Train quantitative models using Microsoft's Qlib
- ⏰ **Automated Data Download**: Daily cron job to fetch latest market data
- 🐳 **Docker Support**: Easy deployment with Docker Compose
- 📈 **Model Management**: Track training history and manage models

## Project Structure

```
qserver/
├── app/                    # Flask web application
│   ├── __init__.py        # Application factory
│   ├── config.py          # Configuration
│   ├── routes.py          # API routes
│   ├── templates/         # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   └── dashboard.html
│   └── static/            # Static files (CSS, JS)
├── qlib_module/           # Qlib training module
│   ├── __init__.py
│   ├── trainer.py         # Model training logic
│   └── data_manager.py    # Data management
├── cron/                  # Cron job scripts
│   ├── download_data.py   # Daily data download
│   ├── train_model.py     # Daily model training
│   └── crontab           # Cron configuration
├── data/                  # Data directory (gitignored)
├── Dockerfile            # Main application Dockerfile
├── Dockerfile.scheduler  # Scheduler Dockerfile
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
├── run.py               # Application entry point
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository and navigate to the project directory:
   ```bash
   cd qserver
   ```

2. Copy the environment file and modify as needed:
   ```bash
   cp .env.example .env
   ```

3. Build and start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at http://localhost:5000

### Local Development (Conda)

1. Create a conda environment:
   ```bash
   conda create -n qserver python=3.10
   conda activate qserver
   ```

2. Install dependencies:
   ```bash
   # Install core packages via conda (recommended for better compatibility)
   conda install -c conda-forge numpy pandas scipy scikit-learn lightgbm xgboost

   # Install remaining packages via pip
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export FLASK_APP=run.py
   export FLASK_ENV=development
   ```

4. Run the application:
   ```bash
   flask run
   # Or: python run.py
   ```

### Local Development (venv)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export FLASK_APP=run.py
   export FLASK_ENV=development
   ```

4. Run the application:
   ```bash
   flask run
   # Or: python run.py
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Dashboard with training status |
| `/api/train` | POST | Start model training |
| `/api/data/status` | GET | Get data status |
| `/api/data/download` | POST | Trigger data download |
| `/api/models` | GET | List trained models |

## Cron Jobs

The following cron jobs are configured:

- **Data Download**: Daily at 00:00 UTC
- **Model Training**: Daily at 02:00 UTC

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `development` |
| `FLASK_DEBUG` | Enable debug mode | `true` |
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `QLIB_DATA_PATH` | Path to Qlib data | `/app/data/qlib_data` |
| `MODEL_OUTPUT_PATH` | Path for saved models | `/app/data/models` |

## Qlib Integration

This project uses [Microsoft Qlib](https://github.com/microsoft/qlib) for:

- Downloading and processing market data
- Training machine learning models (LightGBM, XGBoost, etc.)
- Backtesting trading strategies

### Supported Models

- LightGBM (default)
- XGBoost
- Custom models via Qlib's model interface

## Development

### Running Tests

```bash
pytest tests/ -v --cov=app --cov=qlib_module
```

### Code Style

```bash
# Format code
black app/ qlib_module/

# Check linting
flake8 app/ qlib_module/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
