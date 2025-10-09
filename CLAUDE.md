# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ColdSnap is a Python framework for serializing machine learning models and their training/testing data together as "frozen" snapshots. The core architecture consists of two main classes that work together:

- **Data**: Handles train/test splits, feature information, and data serialization
- **Model**: Contains a scikit-learn estimator (classifier, regressor, or transformer) and associated Data instance, with evaluation capabilities

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
- Combines Data instance with any scikit-learn estimator (classifiers, regressors, transformers)
- Accepts both `clf=` (backward compatible) and `estimator=` parameters
- Automatically detects estimator type and enables appropriate methods
- Inherits evaluation mixins: ConfusionMatrixMixin, ROCMixin, SHAPMixin (classifier-only)
- **Classifiers**: Provides `.fit()`, `.predict()`, `.predict_proba()`, `.evaluate()` methods
- **Regressors**: Provides `.fit()`, `.predict()` methods (evaluation not yet implemented)
- **Transformers**: Provides `.fit()`, `.transform()` methods
- Generates model hash based on serialized estimator
- `.summary()` returns comprehensive model + data metadata (includes estimator type)

**Serializable Base** (`src/coldsnap/serializable.py`)
- Provides `.to_pickle()` and `.from_pickle()` class methods
- Uses gzip compression for efficient storage
- Includes error handling and type validation

### Mixins System

Model evaluation capabilities are provided through mixins in `src/coldsnap/mixins/model_metrics.py`:
- **ConfusionMatrixMixin**: `.confusion_matrix()`, `.display_confusion_matrix()`
- **ROCMixin**: `.display_roc_curve()` with multi-class support
- **SHAPMixin**: `.display_shap_beeswarm()` for feature importance

**Note**: Mixins only work with classifiers and will raise helpful TypeErrors if called on regressors or transformers.

## Usage Examples

### Working with Classifiers (Original Functionality)

```python
from sklearn.ensemble import RandomForestClassifier
from coldsnap import Data, Model

# Create data
data = Data.from_df(df, label_col="target", test_size=0.2, random_state=42)

# Create model with classifier (both syntaxes work)
model1 = Model(data=data, clf=RandomForestClassifier())  # Legacy syntax
model2 = Model(data=data, estimator=RandomForestClassifier())  # New syntax

# Fit, predict, evaluate
model1.fit()
predictions = model1.predict(data.X_test)
metrics = model1.evaluate()

# Visualization methods work for classifiers
model1.display_confusion_matrix()
model1.display_roc_curve()
```

### Working with Transformers (New)

```python
from sklearn.preprocessing import StandardScaler
from coldsnap import Data, Model

# Create data
data = Data.from_df(df, label_col="target", test_size=0.2, random_state=42)

# Create model with transformer
scaler_model = Model(data=data, estimator=StandardScaler())

# Fit and transform
scaler_model.fit()
X_scaled = scaler_model.transform(data.X_train)

# DataFrame structure is automatically preserved!
# X_scaled maintains the same index and column names as data.X_train
print(X_scaled.index)  # Original index preserved
print(X_scaled.columns)  # Original column names preserved

# Save the fitted transformer with data
scaler_model.to_pickle("scaler_snapshot.pkl.gz")

# Load and use later
loaded_scaler = Model.from_pickle("scaler_snapshot.pkl.gz")
X_new_scaled = loaded_scaler.transform(X_new)
```

**DataFrame Preservation**: When you pass a pandas DataFrame to `transform()`, the output automatically preserves:
- **Row index**: Original DataFrame indices are maintained
- **Column names**: Original column headers are preserved (for transformers that don't change column count like StandardScaler, MinMaxScaler)
- **Column tracking**: Even transformers that change columns (PCA, PolynomialFeatures) return DataFrames with appropriate column names

This feature uses scikit-learn's `set_output` API (available since v1.5.2+) and works with all standard transformers.

### Working with Regressors (New)

```python
from sklearn.linear_model import LinearRegression
from coldsnap import Data, Model

# Create regression data
data = Data.from_df(df, label_col="price", test_size=0.2, random_state=42)

# Create model with regressor
reg_model = Model(data=data, estimator=LinearRegression())

# Fit and predict
reg_model.fit()
predictions = reg_model.predict(data.X_test)

# Note: evaluate() not yet implemented for regressors
# model.evaluate()  # Would raise NotImplementedError
```

### Backward Compatibility

All existing code using `clf=` parameter continues to work without modification:

```python
# Old pickles load correctly with zero migration
old_model = Model.from_pickle("old_classifier.pkl.gz")
old_model.fit()  # Works exactly as before

# Both properties are available
assert old_model.clf is old_model.estimator  # True
```

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
- when writing a commit message (for git) don't include emoji, don't mention Claude, Claude code or Antropic
- Don't use emojis in descriptions or titles for github tags and releases