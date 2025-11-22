"""File I/O utilities for TNM staging workflow."""

import json
import logging
from typing import Any, Dict

import pandas as pd


logger = logging.getLogger(__name__)


def load_tnm_json(file_path: str = 'tnm_classification.json') -> Dict[str, Any]:
    """Load TNM classification JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing TNM classification data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Error loading TNM JSON file: {e}")
        raise


def read_excel_data(file_path: str) -> pd.DataFrame:
    """Read Excel or CSV file and return DataFrame.

    Supports both .xlsx (Excel) and .csv (CSV) file formats.
    
    For Excel files, sheet selection follows this priority:
    1. 'data' sheet (if exists) - preferred for compatibility
    2. First sheet (if 'data' sheet not found)
    3. Raises error if no sheets available
    
    This allows flexibility: files with 'data' sheet work as before,
    but files with any other sheet name (e.g., 'Sheet1', 'Cases', etc.)
    will also work by using the first available sheet.

    Args:
        file_path: Path to the Excel (.xlsx) or CSV (.csv) file

    Returns:
        DataFrame containing the file data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read, format is not supported, or no sheets found
    """
    import os
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # Read CSV file
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"{len(df)} records read from CSV file: {file_path}")
            return df
        elif file_ext in ['.xlsx', '.xls']:
            # Read Excel file
            # Try multiple sheet names in order of priority
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            available_sheets = excel_file.sheet_names
            
            # Priority order: 'data' > first sheet > any sheet
            sheet_to_use = None
            if 'data' in available_sheets:
                sheet_to_use = 'data'
            elif len(available_sheets) > 0:
                sheet_to_use = available_sheets[0]
            else:
                raise ValueError(f"No sheets found in Excel file: {file_path}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_to_use, engine='openpyxl')
            logger.info(
                f"{len(df)} records read from Excel file: {file_path} "
                f"(sheet: '{sheet_to_use}', available sheets: {available_sheets})"
            )
            return df
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: .csv, .xlsx, .xls"
            )
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"File reading error: {str(e)}")
        raise

