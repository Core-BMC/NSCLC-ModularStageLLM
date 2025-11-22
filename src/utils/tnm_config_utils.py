"""TNM configuration utilities.

This module provides functions for filtering and formatting TNM-related
configuration data.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def filter_special_notes(special_notes: List[Dict], category: str) -> List[str]:
    """Filter and format special notes based on the specified category.

    Args:
        special_notes: List of special note dictionaries
        category: Category to filter by (e.g., "T category", "N category",
                  "M category")

    Returns:
        List of formatted note strings that apply to the specified category
    """
    try:
        filtered_notes = []
        for note in special_notes:
            if not isinstance(note, dict):
                continue

            applies_to = note.get("applies_to", [])
            note_text = note.get("note")

            if category in applies_to and note_text:
                filtered_notes.append(note_text)

        return filtered_notes

    except Exception as e:
        logger.error(f"Error filtering special notes: {str(e)}")
        return []

