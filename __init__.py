"""Academic Embedding Model Evaluator Package.

This package provides tools for evaluating embedding models on academic paper similarity tasks.
"""

from .config import Config
from .models import ModelManager
from .data import DataManager
from .evaluation import Evaluator
from .utils import console, setup_logging

__version__ = '0.1.0'
