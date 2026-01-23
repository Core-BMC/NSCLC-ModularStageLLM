"""Data processing utilities for TNM staging workflow."""

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd

if TYPE_CHECKING:
    from src.models import Config, InputData, MedicalReport


logger = logging.getLogger(__name__)


def convert_histology_to_structure(histology_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert detailed histology JSON into a simplified structure format.

    Args:
        histology_json: Dictionary containing histology classification data

    Returns:
        Simplified structure dictionary
    """
    logger.debug(f"Converting histology JSON: {histology_json}")
    
    def extract_structure(category_data):
        """Extract structure from category data."""
        if not category_data:
            logger.warning("Empty category data received")
            return {"names": [], "types": []}
            
        structure = {
            "names": [category_data.get("description", "")],
            "types": []
        }
        
        # Handle subcategories case
        if "subcategories" in category_data:
            for subcat_name, subcat_data in category_data["subcategories"].items():
                substructure = extract_structure(subcat_data)
                structure["types"].append({
                    subcat_name: substructure
                })
                
        # Handle types case
        elif "types" in category_data:
            for type_data in category_data["types"]:
                if "subtypes" in type_data:
                    # Handle cases with subtypes
                    substructure = {
                        "names": [type_data["name"]],
                        "subtypes": [
                            subtype["name"] for subtype in type_data["subtypes"]
                        ]
                    }
                    structure["types"].append({
                        type_data["name"].lower().replace(" ", "_"): substructure
                    })
                else:
                    # Simple type case
                    structure["types"].append(type_data["name"])
                    
        return structure
    
    structure = {"categories": {}}
    categories = histology_json.get("categories", {})
    
    if not categories:
        logger.error("No categories found in histology JSON")
        return structure
        
    for category_name, category_data in categories.items():
        structure["categories"][category_name] = extract_structure(category_data)
    
    logger.debug(f"Converted structure: {structure}")
    return structure


def _resolve_hospital_id(row: Dict[str, Any]) -> str:
    """Resolve hospital ID from common column names."""
    candidate_keys = [
        "hospital_id",
        "hospitalNumber",
        "hospital_number",
        "patient_id",
        "patientId",
        "patientID",
        "병원등록번호",
        "병원 등록번호"
    ]

    for key in candidate_keys:
        if key in row:
            value = row.get(key)
            if value is None or pd.isna(value):
                continue
            value_str = str(value).strip()
            if value_str and value_str.lower() != "nan":
                return value_str
    return ""


def format_histology_structure_for_prompt(
    histology_structure: Dict[str, Any],
    max_depth: int = 3
) -> str:
    """Format histology structure for prompt with minimal token usage.
    
    Extracts only category/subcategory/type names in a compact format,
    removing unnecessary descriptions and nested structures.
    
    Args:
        histology_structure: Structure dictionary from convert_histology_to_structure
        max_depth: Maximum depth to include (1=category, 2=subcategory, 3=type)
    
    Returns:
        Compact string representation of the structure
    """
    if not histology_structure or "categories" not in histology_structure:
        return "No classification structure available"
    
    lines = []
    categories = histology_structure.get("categories", {})
    
    for cat_name, cat_data in categories.items():
        # Category level
        cat_display = cat_name.replace("_", " ").title()
        lines.append(f"{cat_display}:")
        
        if max_depth >= 2 and "types" in cat_data:
            for type_item in cat_data["types"]:
                if isinstance(type_item, dict):
                    # Subcategory case
                    for subcat_name, subcat_data in type_item.items():
                        subcat_display = subcat_name.replace("_", " ").title()
                        lines.append(f"  - {subcat_display}")
                        
                        if max_depth >= 3 and isinstance(subcat_data, dict):
                            if "types" in subcat_data:
                                for type_name in subcat_data["types"]:
                                    if isinstance(type_name, str):
                                        lines.append(f"    • {type_name}")
                                    elif isinstance(type_name, dict):
                                        # Handle nested type structures
                                        for nested_key in type_name.keys():
                                            lines.append(f"    • {nested_key.replace('_', ' ').title()}")
                elif isinstance(type_item, str):
                    # Direct type case
                    lines.append(f"  • {type_item}")
        
        lines.append("")  # Empty line between categories
    
    result = "\n".join(lines).strip()
    
    # Log size reduction
    original_size = len(json.dumps(histology_structure, ensure_ascii=False))
    formatted_size = len(result)
    reduction = ((original_size - formatted_size) / original_size * 100) if original_size > 0 else 0
    
    logger.debug(
        f"Formatted structure: {original_size} → {formatted_size} chars "
        f"({reduction:.1f}% reduction)"
    )
    
    return result


def validate_input(
    row: Dict[str, Any],
    case_number: int
) -> 'InputData':
    """Validate input data and create InputData instance.

    Args:
        row: Dictionary containing input data
        case_number: Case number identifier

    Returns:
        Validated InputData instance

    Raises:
        ValueError: If validation fails
    """
    # Import here to avoid circular import
    from src.models import InputData, MedicalReport
    
    try:
        input_data = {
            "case_number": case_number,
            "hospital_id": _resolve_hospital_id(row),
            "pathology": (
                MedicalReport(content=row.get("Pathology"))
                if not pd.isna(row.get("Pathology")) else None
            ),
            "chest_ct": (
                MedicalReport(content=row.get("Chest CT"))
                if not pd.isna(row.get("Chest CT")) else None
            ),
            "brain_mr": (
                MedicalReport(content=row.get("Brain MR"))
                if not pd.isna(row.get("Brain MR")) else None
            ),
            "pet": (
                MedicalReport(content=row.get("PET"))
                if not pd.isna(row.get("PET")) else None
            ),
            "ebus": (
                MedicalReport(content=row.get("EBUS"))
                if not pd.isna(row.get("EBUS")) else None
            ),
            "neck_biopsy": (
                MedicalReport(content=row.get("neck biopsy"))
                if not pd.isna(row.get("neck biopsy")) else None
            ),
            "bone_scan": (
                MedicalReport(content=row.get("Bone scan"))
                if not pd.isna(row.get("Bone scan")) else None
            ),
            "abdomen_pelvis_ct": (
                MedicalReport(content=row.get("Abdomen&Pelvis CT"))
                if not pd.isna(row.get("Abdomen&Pelvis CT")) else None
            ),
            "adrenal_ct": (
                MedicalReport(content=row.get("Adrenal CT"))
                if not pd.isna(row.get("Adrenal CT")) else None
            )
        }
        validated_data = InputData(**input_data)
        logger.info(f"Input data validated for case {case_number}")
        return validated_data
    except Exception as e:
        logger.error(
            f"Input data validation failed for case {case_number}: {str(e)}"
        )
        raise ValueError(
            f"Invalid input data for case {case_number}: {str(e)}"
        )


def prepare_input(
    row: Dict[str, Any],
    case_number: int
) -> Dict[str, Any]:
    """Prepare input data for model processing.

    Args:
        row: Dictionary containing input data
        case_number: Case number identifier

    Returns:
        Dictionary with 'model_input' and 'true_tnm' keys

    Raises:
        ValueError: If data preparation fails
    """
    # Import here to avoid circular import
    from src.models import InputData, MedicalReport
    
    try:
        # Validate data excluding true TNM fields from InputData class
        input_data = {
            "case_number": case_number,
            "hospital_id": _resolve_hospital_id(row),
            "pathology": (
                MedicalReport(content=row.get("Pathology"))
                if not pd.isna(row.get("Pathology")) else None
            ),
            "chest_ct": (
                MedicalReport(content=row.get("Chest CT"))
                if not pd.isna(row.get("Chest CT")) else None
            ),
            "brain_mr": (
                MedicalReport(content=row.get("Brain MR"))
                if not pd.isna(row.get("Brain MR")) else None
            ),
            "pet": (
                MedicalReport(content=row.get("PET"))
                if not pd.isna(row.get("PET")) else None
            ),
            "ebus": (
                MedicalReport(content=row.get("EBUS"))
                if not pd.isna(row.get("EBUS")) else None
            ),
            "neck_biopsy": (
                MedicalReport(content=row.get("neck biopsy"))
                if not pd.isna(row.get("neck biopsy")) else None
            ),
            "bone_scan": (
                MedicalReport(content=row.get("Bone scan"))
                if not pd.isna(row.get("Bone scan")) else None
            ),
            "abdomen_pelvis_ct": (
                MedicalReport(content=row.get("Abdomen&Pelvis CT"))
                if not pd.isna(row.get("Abdomen&Pelvis CT")) else None
            ),
            "adrenal_ct": (
                MedicalReport(content=row.get("Adrenal CT"))
                if not pd.isna(row.get("Adrenal CT")) else None
            ),
        }
        validated_data = InputData(**input_data)
        processed_data = validated_data.model_dump(
            exclude_none=True,
            exclude_unset=True
        )

        # Store True TNM data separately
        true_tnm_data = {
            'true_t': row.get('cT'),
            'true_n': row.get('cN'),
            'true_m': row.get('cM'),
            'true_stage': row.get('cStage')
        }

        # Process medical report data
        for key, value in processed_data.items():
            if isinstance(value, dict) and 'content' in value:
                processed_data[key] = value['content']

        result = {
            "model_input": processed_data,  # Exclude True TNM data
            "true_tnm": true_tnm_data      # Store separately
        }
        
        logger.info(
            f"Case {case_number} input data preparation completed"
        )
        return result
        
    except Exception as e:
        logger.error(
            f"Case {case_number} input data preparation failed: {str(e)}"
        )
        raise


def process_excel_data(config: 'Config') -> List[Dict[str, Any]]:
    """Process Excel or CSV file and prepare data for workflow.

    Supports both .xlsx (Excel) and .csv (CSV) file formats.

    Args:
        config: Configuration object

    Returns:
        List of dictionaries containing processed case data

    Raises:
        ValueError: If input file is not configured
    """
    from src.utils.file_utils import read_excel_data
    
    input_file = config.get('input_file')
    if not input_file:
        logger.error("Input file is not configured.")
        raise ValueError("Missing input file configuration")

    df = read_excel_data(input_file)
    processed_data = []
    
    for index, row in df.iterrows():
        try:
            case_number = index + 1
            prepared_data = prepare_input(row.to_dict(), case_number)
            
            # Separate and store model input and True TNM data
            processed_data.append({
                "input": prepared_data["model_input"],
                "case_number": case_number,
                "true_tnm": prepared_data["true_tnm"]
            })
            
        except Exception as e:
            logger.error(
                f"Case {case_number} error occurred during processing: "
                f"{str(e)}"
            )
            continue
    
    return processed_data


def format_input_data(data: Dict[str, Any]) -> str:
    """Format input data as readable text preserving structure.

    Args:
        data: Dictionary containing input data with case_number, hospital_id,
              and various medical reports

    Returns:
        Formatted string representation of the input data
    """
    try:
        formatted = []

        # Add basic info
        case_number = data.get('case_number')
        hospital_id = data.get('hospital_id')

        if case_number is not None:
            formatted.append(f"Case Number: {case_number}")
        if hospital_id:
            formatted.append(f"Hospital ID: {hospital_id}")

        # Add reports in a structured way
        # Handle both dict format {'content': '...'} and direct string format
        for key, value in data.items():
            # Skip metadata fields
            if key in ['case_number', 'hospital_id', 'histology_classification', 
                      't_classification', 'n_classification', 'm_classification', 
                      'stage_classification']:
                continue
            
            # Handle dict format with 'content' key
            if isinstance(value, dict) and 'content' in value:
                content = value['content']
                if content:
                    formatted.append(f"\n{key.upper().replace('_', ' ')}:")
                    formatted.append(str(content))
            # Handle direct string format
            elif isinstance(value, str) and value.strip():
                formatted.append(f"\n{key.upper().replace('_', ' ')}:")
                formatted.append(value)
            # Handle MedicalReport objects
            elif hasattr(value, 'content'):
                if value.content:
                    formatted.append(f"\n{key.upper().replace('_', ' ')}:")
                    formatted.append(str(value.content))

        result = "\n".join(formatted)
        # Log formatted data for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Formatted input data (first 1000 chars): {result[:1000]}")
        return result

    except Exception as e:
        logger.error(f"Error formatting input data: {str(e)}")
        return str(data)  # Fallback to simple string conversion
