# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-07

### Added
- Support for sklearn transformers (StandardScaler, PCA, MinMaxScaler, etc.)
- Support for sklearn regressors (LinearRegression, Ridge, Lasso, etc.)
- New `estimator` parameter for Model class (more intuitive than `clf`)
- `transform()` method for transformer workflows
- Type detection system to distinguish between classifiers, regressors, and transformers
- Comprehensive error handling with helpful TypeError messages for incompatible operations
- 100% test coverage across all source files
- Transformer usage example in README

### Changed
- Model class now supports three types of estimators: classifiers, regressors, and transformers
- `_get_estimator_type()` method returns estimator type as string
- Summary now includes `estimator_type` field
- Enhanced documentation in CLAUDE.md with usage examples for all estimator types

### Fixed
- Type safety: mixins now properly reject non-classifier estimators
- Proper error messages when calling incompatible methods (e.g., predict on transformer)

### Deprecated
- None (full backward compatibility maintained)

### Removed
- None

### Security
- None

### Notes
- The `clf` parameter is still fully supported for backward compatibility
- All existing pickled models from v0.0.2 load without any changes required
- Evaluation for regressors is not yet implemented (raises NotImplementedError)

## [0.0.2]

### Added
- Initial public release
- Data class for train/test split management
- Model class for classifier serialization
- Confusion matrix visualization
- ROC curve visualization
- SHAP beeswarm plot support
- Model evaluation metrics (accuracy, precision, recall, f1, roc_auc)
- `purge()` method to remove data while keeping headers
- `create_overview()` utility for model comparison
