"""Configuration and data loading utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _resolve_path(path: str, base_path: Optional[str] = None) -> str:
    """Resolve relative or absolute path.

    Args:
        path: Path to resolve (can be relative or absolute)
        base_path: Base path for relative path resolution (defaults to current working directory)

    Returns:
        Absolute path string
    """
    if os.path.isabs(path):
        return path

    # If path starts with 'config/', resolve from project root
    if path.startswith('config/'):
        project_root = Path.cwd()
        resolved = (project_root / path).resolve()
        if resolved.exists():
            return str(resolved)

    if base_path:
        # If base_path is a file, use its parent directory
        if os.path.isfile(base_path):
            base = Path(base_path).parent
        else:
            base = Path(base_path)
    else:
        base = Path.cwd()

    resolved = (base / path).resolve()
    return str(resolved)


def load_config(config_path: str, tnm_json_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file
        tnm_json_path: Optional path to TNM JSON file (overrides config setting)

    Returns:
        Configuration dictionary with loaded TNM data

    Raises:
        Exception: If configuration loading fails
    """
    try:
        # Resolve config path to absolute path
        config_path_abs = _resolve_path(config_path)
        logger.debug(f"Loading config from: {config_path_abs}")

        with open(config_path_abs, 'r', encoding='utf-8') as file:
            config_yaml = yaml.safe_load(file)

        # Get tnm_json_path from config if not provided
        if not tnm_json_path:
            tnm_json_path = config_yaml.get('tnm_json', 'tnm_classification.json')

        # Resolve TNM JSON path
        # If path starts with 'config/', resolve from project root
        # Otherwise, resolve relative to config file location
        if tnm_json_path.startswith('config/'):
            project_root = Path.cwd()
            tnm_json_path_abs = str((project_root / tnm_json_path).resolve())
        else:
            tnm_json_path_abs = _resolve_path(tnm_json_path, config_path_abs)
        
        logger.info(f"Loading TNM JSON from: {tnm_json_path_abs}")

        # Detect AJCC edition from tnm_json path (check both original and absolute paths)
        ajcc_edition = None
        ajcc_version = None
        # Check both original path and absolute path for edition detection
        path_to_check = tnm_json_path.lower() + ' ' + tnm_json_path_abs.lower()
        if 'ajcc8th' in path_to_check:
            ajcc_edition = 'ajcc8th'
            ajcc_version = '8th'
        elif 'ajcc9th' in path_to_check:
            ajcc_edition = 'ajcc9th'
            ajcc_version = '9th'
        else:
            logger.warning(
                f"Could not detect AJCC edition from path: {tnm_json_path} (abs: {tnm_json_path_abs}). "
                f"Defaulting to ajcc8th"
            )
            ajcc_edition = 'ajcc8th'
            ajcc_version = '8th'

        # Store resolved path and detected edition in config
        config_yaml['tnm_json_path'] = tnm_json_path_abs
        config_yaml['tnm_json'] = tnm_json_path_abs
        config_yaml['ajcc_edition'] = ajcc_edition
        config_yaml['ajcc_version'] = ajcc_version
        
        logger.info(
            f"Detected AJCC edition: {ajcc_edition} "
            f"(AJCC {ajcc_version} edition)"
        )

        # Load and save TNM JSON data
        tnm_data = load_tnm_json(tnm_json_path_abs)
        config_yaml['tnm_criteria'] = tnm_data
        config_yaml['histology_criteria'] = tnm_data.get(
            'histology_classification', {}
        )
        config_yaml['histology_rules'] = tnm_data.get('histology_rules', [])

        # Store stage_rules from config
        config_yaml['stage_rules'] = config_yaml.get('stage_rules', {})

        # Store config file directory for reference
        config_yaml['config_dir'] = str(Path(config_path_abs).parent)

        return config_yaml
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def load_tnm_json(tnm_json_path: str) -> Dict[str, Any]:
    """Load TNM classification JSON file.

    Args:
        tnm_json_path: Path to TNM JSON file (can be relative or absolute)

    Returns:
        TNM data dictionary

    Raises:
        FileNotFoundError: If JSON file is not found
        json.JSONDecodeError: If JSON parsing fails
        Exception: If other errors occur
    """
    try:
        # Resolve path to absolute path
        tnm_json_path_abs = _resolve_path(tnm_json_path)
        logger.debug(f"Loading TNM JSON from: {tnm_json_path_abs}")

        if not os.path.exists(tnm_json_path_abs):
            # Try common locations if file not found
            project_root = Path.cwd()
            possible_paths = [
                tnm_json_path_abs,
                project_root / tnm_json_path,
                project_root / 'config' / tnm_json_path,
                project_root / 'config' / 'ajcc8th' / 'tnm_classification.json',
                project_root / 'config' / 'ajcc9th' / 'tnm_classification.json',
            ]

            for path in possible_paths:
                path_str = str(path) if isinstance(path, Path) else path
                if os.path.exists(path_str):
                    tnm_json_path_abs = path_str
                    logger.info(f"Found TNM JSON at: {tnm_json_path_abs}")
                    break
            else:
                raise FileNotFoundError(
                    f"TNM JSON file not found: {tnm_json_path}. "
                    f"Tried paths: {[str(p) for p in possible_paths]}"
                )

        with open(tnm_json_path_abs, 'r', encoding='utf-8') as file:
            tnm_data = json.load(file)
            logger.info(
                f"Successfully loaded TNM data from {tnm_json_path_abs} "
                f"with keys: {list(tnm_data.keys())}"
            )
            return tnm_data
    except FileNotFoundError:
        logger.error(f"TNM JSON file not found: {tnm_json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {tnm_json_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading TNM JSON file: {e}")
        raise


def load_all_tnm_jsons(
    config_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Load all available TNM JSON files (ajcc8th and ajcc9th).

    Args:
        config_dir: Base directory for searching JSON files
                    (defaults to current working directory)

    Returns:
        Dictionary with keys 'ajcc8th' and 'ajcc9th', each containing
        the loaded TNM data, or empty dict if file not found

    Example:
        >>> tnm_data = load_all_tnm_jsons()
        >>> ajcc8th_data = tnm_data.get('ajcc8th', {})
        >>> ajcc9th_data = tnm_data.get('ajcc9th', {})
    """
    if config_dir:
        base_dir = Path(config_dir)
    else:
        base_dir = Path.cwd()

    result = {}
    json_files = {
        'ajcc8th': base_dir / 'config' / 'ajcc8th' / 'tnm_classification.json',
        'ajcc9th': base_dir / 'config' / 'ajcc9th' / 'tnm_classification.json',
    }

    for edition, json_path in json_files.items():
        try:
            if json_path.exists():
                result[edition] = load_tnm_json(str(json_path))
                logger.info(f"Loaded {edition} TNM data")
            else:
                logger.warning(f"{edition} JSON file not found: {json_path}")
        except Exception as e:
            logger.warning(f"Failed to load {edition} JSON: {e}")

    return result


def setup_environment():
    """Load environment variables."""
    load_dotenv()


def setup_user_paths(
    user_id: Optional[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Set user-specific input/output paths.

    Args:
        user_id: Optional user ID
        config: Configuration dictionary to update

    Returns:
        Updated configuration dictionary
    """
    base_output = os.path.abspath(os.path.join(os.getcwd(), 'output'))

    if user_id:
        user_output = os.path.join(base_output, f'user_{user_id}')
        os.makedirs(user_output, exist_ok=True)

        config['output_file'] = os.path.join(
            user_output, f'tnm_classification_{user_id}.csv'
        )
        config['json_output_file'] = os.path.join(
            user_output, f'patient_data_{user_id}.json'
        )
        config['log_file'] = os.path.join(
            user_output, f'tnm_classification_{user_id}.log'
        )

        logger.info(f"Set up user paths for user {user_id}:")
        logger.info(f"Output file: {config['output_file']}")
        logger.info(f"JSON file: {config['json_output_file']}")

    else:
        # Use default paths when no user ID provided
        os.makedirs(base_output, exist_ok=True)
        config['output_file'] = os.path.join(base_output, 'tnm_classification.csv')
        config['json_output_file'] = os.path.join(base_output, 'patient_data.json')
        config['log_file'] = os.path.join(base_output, 'tnm_classification.log')

        logger.warning("No user ID provided, using default paths")

    return config

