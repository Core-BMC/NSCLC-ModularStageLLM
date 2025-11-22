"""File operations for saving classification results."""

import csv
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def save_to_json_file(
    classification: Dict[str, Any],
    input_data: Dict[str, Any],
    json_file_path: str
):
    """Save classification results to JSON file.

    Args:
        classification: Classification result dictionary
        input_data: Input data dictionary
        json_file_path: Path to JSON output file

    Raises:
        Exception: If saving fails
    """
    try:
        # Get true TNM data directly from the classification
        true_tnm = classification.get('true_tnm', {})

        logger.debug(f"True TNM data before saving: {true_tnm}")

        # Prepare histology info
        histology_info = {
            "category": classification.get('histology_category'),
            "subcategory": classification.get('histology_subcategory', ''),
            "type": classification.get('histology_type'),
            "confidence": classification.get('histology_confidence'),
            "reasoning": classification.get('histology_reason')
        }

        # Get unified TNM classifications
        classifications = classification.get('classifications', {})
        
        # Get reasonings
        reasonings = classification.get('reasonings', {})

        # Check if true_tnm is not string 'null' or None
        true_t = (
            true_tnm.get('true_t')
            if true_tnm.get('true_t') not in [None, 'null', '']
            else None
        )
        true_n = (
            true_tnm.get('true_n')
            if true_tnm.get('true_n') not in [None, 'null', '']
            else None
        )
        true_m = (
            true_tnm.get('true_m')
            if true_tnm.get('true_m') not in [None, 'null', '']
            else None
        )
        true_stage = (
            true_tnm.get('true_stage')
            if true_tnm.get('true_stage') not in [None, 'null', '']
            else None
        )

        patient_data = {
            "id": classification['case_number'],
            "hospitalNumber": input_data.get('hospital_id'),
            "histology": histology_info,
            "reports": {
                "Pathology": input_data.get('pathology') or '',
                "Chest CT": input_data.get('chest_ct') or '',
                "Brain MR": input_data.get('brain_mr') or '',
                "PET": input_data.get('pet') or '',
                "EBUS": input_data.get('ebus') or '',
                "neck biopsy": input_data.get('neck_biopsy') or '',
                "Bone scan": input_data.get('bone_scan') or '',
                "Abdomen&Pelvis CT": input_data.get('abdomen_pelvis_ct') or '',
                "Adrenal CT": input_data.get('adrenal_ct') or ''
            },
            "tnm": {
                "T": true_t,
                "N": true_n,
                "M": true_m,
                "Stage": true_stage
            },
            "aiTnm": {
                "T": classifications.get('t_classification'),
                "N": classifications.get('n_classification'),
                "M": classifications.get('m_classification'),
                "Stage": classifications.get('stage_classification')
            },
            "reasonings": {
                "T": reasonings.get('t_reasoning', ''),
                "N": reasonings.get('n_reasoning', ''),
                "M": reasonings.get('m_reasoning', ''),
                "Stage": reasonings.get('stage_reasoning', '')
            },
            "raw_outputs": {
                "T": input_data.get('t_classification', {}).get('raw_output', ''),
                "N": input_data.get('n_classification', {}).get('raw_output', ''),
                "M": input_data.get('m_classification', {}).get('raw_output', ''),
                "Stage": input_data.get('stage_classification', {}).get('raw_output', '') or 'Rule-based calculation (no LLM response)'
            }
        }

        logger.debug(f"Patient data TNM before saving: {patient_data['tnm']}")

        # Handle existing data
        existing_data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []

        existing_data.append(patient_data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

        # Write updated data
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Successfully saved patient data (ID: {patient_data['id']}) "
            f"to JSON file: {json_file_path}"
        )

    except Exception as e:
        logger.error(f"Error saving to JSON file: {e}", exc_info=True)
        raise


def save_to_csv_file(
    classification: Dict[str, Any],
    output_file: str,
    fieldnames: list
):
    """Save classification results to CSV file.

    Args:
        classification: Classification result dictionary
        output_file: Path to CSV output file
        fieldnames: List of CSV column names

    Raises:
        Exception: If saving fails
    """
    try:
        if not output_file:
            raise ValueError("Output file path not configured")

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load existing data
        existing_rows = []
        if os.path.exists(output_file):
            with open(
                output_file, 'r', newline='', encoding='utf-8'
            ) as csvfile:
                reader = csv.DictReader(csvfile)
                existing_rows = list(reader)

        # Generate new PID
        pid = len(existing_rows) + 1

        # Get classifications from unified structure
        classifications = classification.get('classifications', {})
        
        # Get reasonings
        reasonings = classification.get('reasonings', {})

        # Get true TNM data from state
        true_tnm = classification.get('true_tnm', {})
        logger.debug(f"True TNM data for saving: {true_tnm}")

        # Create new row
        new_row = {
            'pid': pid,
            'hospital_number': classification.get('input', {}).get(
                'hospital_id', ''
            ),
            'histology_category': classification.get('histology_category', ''),
            'histology_subcategory': classification.get(
                'histology_subcategory', ''
            ),
            'histology_type': classification.get('histology_type', ''),
            'histology_confidence': classification.get(
                'histology_confidence', ''
            ),
            'histology_reason': classification.get('histology_reason', ''),
            # True TNM values
            'true_T': true_tnm.get('true_t', ''),
            'true_N': true_tnm.get('true_n', ''),
            'true_M': true_tnm.get('true_m', ''),
            'true_Stage': true_tnm.get('true_stage', ''),
            # AI predictions
            'T_classification': classifications.get('t_classification', ''),
            'N_classification': classifications.get('n_classification', ''),
            'M_classification': classifications.get('m_classification', ''),
            'Stage_classification': classifications.get(
                'stage_classification', ''
            ),
            # Reasoning fields
            'T_reasoning': reasonings.get('t_reasoning', ''),
            'N_reasoning': reasonings.get('n_reasoning', ''),
            'M_reasoning': reasonings.get('m_reasoning', ''),
            'Stage_reasoning': reasonings.get('stage_reasoning', '')
        }

        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            if existing_rows:
                writer.writerows(existing_rows)
            writer.writerow(new_row)

        logger.info(f"Successfully wrote data to CSV file: {output_file}")

    except Exception as e:
        logger.error(f"Error writing to CSV file: {e}", exc_info=True)
        raise


def finalize_csv(output_file: str, fieldnames: list):
    """Finalize CSV file creation.

    Args:
        output_file: Path to CSV output file
        fieldnames: List of CSV column names

    Raises:
        Exception: If finalization fails
    """
    try:
        if not output_file:
            logger.error("Output file path not configured")
            raise ValueError("Output file path not configured")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(output_file):
            logger.warning(
                f"No data was written to CSV file: {output_file}"
            )
            # Create empty CSV with headers
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        logger.info(f"Successfully finalized CSV file: {output_file}")

    except Exception as e:
        logger.error(f"Error finalizing CSV file: {e}", exc_info=True)
        raise

