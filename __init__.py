"""
ACC Base: Advanced Classification and Clustering Base Framework.

A PyTorch Lightning-based framework for time series classification and analysis.
"""

__version__ = "1.0.0"
__author__ = "ACC Team"

try:
    from pipeline.classification import ClassificationPipeline
    from pipeline.mae import MAEPipeline
    from pipeline.simclr import SimCLRPipeline

    __all__ = [
        "ClassificationPipeline",
        "MAEPipeline",
        "SimCLRPipeline",
        "create_pipeline"
    ]

except ImportError:
    # Handle case where pipeline modules aren't available (e.g., during testing setup)
    __all__ = []