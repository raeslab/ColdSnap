# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ColdSnap is a Python framework for serializing machine learning models and their training/testing data together as "frozen" snapshots. The core architecture consists of two main classes that work together:

- **Data**: Handles train/test splits, feature information, and data serialization
- **Model**: Contains a scikit-learn classifier and associated Data instance, with evaluation capabilities

Both classes inherit from `Serializable` which provides pickle-based persistence with gzip compression.

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest --exitfirst --verbose --failed-first --cov=src tests/ --cov-report=term-missing

# Run specific test file
pytest tests/test_model.py -v

# Run single test
pytest tests/test_model.py::TestModel::test_fit -v
```

### Code Quality
```bash
# Check code formatting and linting
ruff check

# Auto-fix formatting issues
ruff check --fix

# Format code
ruff format
```

### Installation
```bash
# Install package in development mode
pip install -e .
```

## Architecture

### Core Classes

**Data Class** (`src/coldsnap/data.py`)
- Wraps train/test splits as pandas DataFrames/Series
- Provides `.from_df()` classmethod for easy creation from single DataFrame
- Includes `.purge()` method to remove data while keeping headers
- Generates SHA256 hash of all data for integrity checking

**Model Class** (`src/coldsnap/model.py`)
- Combines Data instance with scikit-learn classifier
- Inherits evaluation mixins: ConfusionMatrixMixin, ROCMixin, SHAPMixin
- Provides `.fit()`, `.predict()`, `.evaluate()` methods
- Generates model hash based on serialized classifier
- `.summary()` returns comprehensive model + data metadata

**Serializable Base** (`src/coldsnap/serializable.py`)
- Provides `.to_pickle()` and `.from_pickle()` class methods
- Uses gzip compression for efficient storage
- Includes error handling and type validation

### Mixins System

Model evaluation capabilities are provided through mixins in `src/coldsnap/mixins/model_metrics.py`:
- **ConfusionMatrixMixin**: `.confusion_matrix()`, `.display_confusion_matrix()`
- **ROCMixin**: `.display_roc_curve()` with multi-class support
- **SHAPMixin**: `.display_shap_beeswarm()` for feature importance

### Utilities

**create_overview()** (`src/coldsnap/utils.py`)
- Takes list of model pickle paths
- Returns pandas DataFrame with summary + evaluation metrics for all models
- Used for model comparison and reporting

## Testing Strategy

- Tests use pytest with fixtures defined in `tests/conftest.py`
- Coverage target maintained through GitHub Actions
- Test structure mirrors source code organization
- Focus on both individual class functionality and integration scenarios

## Package Structure

```
src/coldsnap/
├── __init__.py          # Main exports: Data, Model
├── data.py              # Data class with train/test handling
├── model.py             # Model class with classifier integration
├── serializable.py      # Base serialization functionality
├── utils.py             # Utility functions like create_overview()
└── mixins/
    ├── __init__.py      # Mixin exports
    └── model_metrics.py # Evaluation and visualization mixins
```

## Development Workflow

1. The project uses automated GitHub Actions for testing (Python 3.10-3.13) and code formatting
2. Ruff is used for both linting and formatting
3. pytest runs with coverage reporting and badge generation
4. Both ruff and pytest actions can auto-commit fixes when needed
- always run ruff check and ruff format to fix linting and formating errors when making major changes to python code (.py files)