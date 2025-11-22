"""Histology classification type definitions."""

from typing import Optional, TypedDict


class HistologyClassification(TypedDict):
    """Type definition for histology classification result."""

    main_category: str
    subcategory: str  # Add subcategory field
    type: str
    confidence: str
    reasoning: str
    raw_text: Optional[str]

