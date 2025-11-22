"""Configuration management class with user-specific paths."""

import json
import logging
import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from src.utils.file_utils import load_tnm_json

logger = logging.getLogger(__name__)


class Config:
    """Configuration management class with user-specific paths."""

    def __init__(
        self,
        config_path="config/tnm_config.yaml",
        tnm_json_path=None,
        user_id=None
    ):
        """Initialize configuration.

        Args:
            config_path: Path to configuration YAML file
            tnm_json_path: Path to TNM classification JSON file
            user_id: Optional user ID for user-specific paths
        """
        self.user_id = user_id
        self.config_path = config_path
        self.config = self.load_config(config_path, tnm_json_path)
        self.tnm_json_path = self.config.get(
            'tnm_json', tnm_json_path or 'tnm_classification.json'
        )
        self.setup_environment()
        self.max_iterations = self.get('max_iterations', 2)
        self.tnm_data = self.load_tnm_json(self.tnm_json_path)

        # Set user-specific paths
        self.setup_user_paths()

    def set_config_paths(self, config_path=None, tnm_json_path=None):
        """Set configuration file paths.

        Args:
            config_path: Path to configuration YAML file
            tnm_json_path: Path to TNM classification JSON file
        """
        if config_path:
            self.config_path = config_path
            self.config = self.load_config(config_path, self.tnm_json_path)
        if tnm_json_path:
            self.tnm_json_path = tnm_json_path
            self.tnm_data = self.load_tnm_json(tnm_json_path)

    def setup_user_paths(self):
        """Set user-specific input/output paths."""
        base_output = os.path.abspath(
            os.path.join(os.getcwd(), 'output')
        )

        if self.user_id:
            user_output = os.path.join(base_output, f'user_{self.user_id}')
            os.makedirs(user_output, exist_ok=True)

            self.config['output_file'] = os.path.join(
                user_output, f'tnm_classification_{self.user_id}.csv'
            )
            self.config['json_output_file'] = os.path.join(
                user_output, f'patient_data_{self.user_id}.json'
            )
            self.config['log_file'] = os.path.join(
                user_output, f'tnm_classification_{self.user_id}.log'
            )

            logger.info(f"Set up user paths for user {self.user_id}:")
            logger.info(f"Output file: {self.config['output_file']}")
            logger.info(f"JSON file: {self.config['json_output_file']}")

        else:
            # Use default paths when no user ID provided
            os.makedirs(base_output, exist_ok=True)
            self.config['output_file'] = os.path.join(
                base_output, 'tnm_classification.csv'
            )
            self.config['json_output_file'] = os.path.join(
                base_output, 'patient_data.json'
            )
            self.config['log_file'] = os.path.join(
                base_output, 'tnm_classification.log'
            )

            logger.warning("No user ID provided, using default paths")

    def get_file_paths(self):
        """Return currently configured file paths.

        Returns:
            Dictionary with file paths
        """
        return {
            'input_file': self.config.get('input_file'),
            'output_file': self.config.get('output_file'),
            'json_output_file': self.config.get('json_output_file'),
            'log_file': self.config.get('log_file')
        }

    def set_input_file(self, file_path):
        """Set input file path.

        Args:
            file_path: Path to input file
        """
        self.config['input_file'] = file_path

    def load_config(self, config_path, tnm_json_path):
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file
            tnm_json_path: Path to TNM classification JSON file

        Returns:
            Configuration dictionary

        Raises:
            Exception: If configuration loading fails
        """
        try:
            # Use utility function for consistent behavior
            from src.utils.config_loader import load_config as load_config_util
            
            config_yaml = load_config_util(config_path, tnm_json_path)
            
            # Load TNM JSON data using the resolved path
            resolved_tnm_json_path = config_yaml.get('tnm_json_path')
            if resolved_tnm_json_path:
                tnm_data = self.load_tnm_json(resolved_tnm_json_path)
            else:
                # Fallback to original path if resolved path not available
                if not tnm_json_path:
                    tnm_json_path = config_yaml.get(
                        'tnm_json', 'tnm_classification.json'
                    )
                tnm_data = self.load_tnm_json(tnm_json_path)
            
            # Ensure TNM data is in config (may already be there from utility)
            config_yaml['tnm_criteria'] = tnm_data
            config_yaml['histology_criteria'] = tnm_data.get(
                'histology_classification', {}
            )
            config_yaml['histology_rules'] = tnm_data.get(
                'histology_rules', []
            )

            return config_yaml
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def load_tnm_json(self, tnm_json_path):
        """Load TNM classification JSON file.

        Args:
            tnm_json_path: Path to TNM JSON file

        Returns:
            TNM classification data dictionary

        Raises:
            Exception: If JSON loading fails
        """
        try:
            with open(tnm_json_path, 'r', encoding='utf-8') as file:
                tnm_data = json.load(file)
                logger.debug(
                    f"Loaded TNM data with keys: {list(tnm_data.keys())}"
                )
                return tnm_data
        except Exception as e:
            logger.error(f"Error loading TNM JSON file: {e}")
            raise

    def get(self, key, default=None):
        """Get configuration value with debug logging.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        value = self.config.get(key, default)
        if key in ['histology_criteria', 'histology_rules']:
            logger.debug(f"Retrieved {key}: {value}")
        return value

    def setup_environment(self):
        """Load environment variables."""
        load_dotenv()

