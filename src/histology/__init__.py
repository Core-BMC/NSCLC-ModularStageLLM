"""Histology classification modules for TNM staging workflow."""

try:
    from src.histology.classification import HistologyClassification
    from src.histology.parser import HistologyOutputParser
    from src.histology.workflow import HistologyClassificationWorkflow
except ImportError:
    # Fallback for relative imports
    from .classification import HistologyClassification
    from .parser import HistologyOutputParser
    from .workflow import HistologyClassificationWorkflow

__all__ = [
    'HistologyClassification',
    'HistologyOutputParser',
    'HistologyClassificationWorkflow',
]

