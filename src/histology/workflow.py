"""Histology classification workflow using LLM agents."""

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

try:
    from src.histology.parser import HistologyOutputParser
    from src.models.config import Config
except ImportError:
    # Fallback for relative imports
    from .parser import HistologyOutputParser
    from ..models.config import Config


logger = logging.getLogger(__name__)


class HistologyClassificationWorkflow:
    """Workflow for histology classification."""

    def __init__(self, config: Config):
        """Initialize histology classification workflow.

        Args:
            config: Configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Load histology classifier prompt from config
        # Check histology_settings first (new location), then fallback to prompts (old location)
        histology_settings = config.get('histology_settings', {})
        self.raw_prompt = histology_settings.get('histology_classifier', '')
        
        # Fallback to old location for backward compatibility
        if not self.raw_prompt:
            prompts = config.get('prompts', {})
            self.raw_prompt = prompts.get('histology_classifier', '')
        
        if not self.raw_prompt:
            self.logger.warning(
                "histology_classifier prompt not found in config "
                "(checked histology_settings.histology_classifier and prompts.histology_classifier), "
                "using default prompt"
            )
            self.raw_prompt = None
        
        # Load histology criteria and rules for prompt formatting
        self.histology_criteria = config.get('histology_criteria', {})
        self.histology_rules = config.get('histology_rules', [])
        
        # Convert histology criteria to structure if needed
        try:
            from src.utils.data_utils import convert_histology_to_structure
            if self.histology_criteria:
                self.histology_structure = convert_histology_to_structure(
                    self.histology_criteria
                )
            else:
                self.histology_structure = {}
        except Exception as e:
            self.logger.warning(
                f"Could not convert histology structure: {e}. "
                f"Using empty structure."
            )
            self.histology_structure = {}
        
        # Setup components
        self._setup_llms()
        self._setup_agents()
        self.logger.info(
            "HistologyClassificationWorkflow initialized successfully"
        )

    def _setup_llms(self):
        """Setup LLM based on configuration."""
        try:
            # Get settings from model_settings
            model_settings = self.config.get('model_settings', {})
            llm_choice = model_settings.get('llm_choice', 'local')
            
            if llm_choice == "local":
                # Load environment variables
                load_dotenv()
                
                # Use local API settings
                local_settings = model_settings.get('local', {})
                
                # Allow environment variables to override YAML settings
                base_url = (
                    os.getenv('LOCAL_API_BASE_URL') or
                    local_settings.get('base_url')
                )
                api_key = (
                    os.getenv('LOCAL_API_KEY') or
                    local_settings.get('api_key')
                )
                model_name = (
                    os.getenv('LOCAL_API_MODEL_NAME') or
                    local_settings.get('name')
                )
                temperature = float(
                    os.getenv('LOCAL_API_TEMPERATURE') or
                    local_settings.get('temperature', 0.0)
                )
                
                # Check if SSL verification should be disabled (for self-signed certificates)
                verify_ssl = local_settings.get('verify_ssl', True)
                if isinstance(verify_ssl, str):
                    verify_ssl = verify_ssl.lower() in ('true', '1', 'yes')
                
                # Ensure base_url ends with /v1 if it doesn't already
                # ChatOpenAI automatically appends /chat/completions, so base_url should be like https://host:port/v1
                if base_url and not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
                    # Check if /v1 is already in the path
                    if '/v1/' not in base_url:
                        base_url = base_url.rstrip('/') + '/v1'
                
                # Create httpx client with SSL verification setting
                http_client = httpx.Client(verify=verify_ssl) if not verify_ssl else None
                
                self.llm = ChatOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    http_client=http_client
                )
                self.logger.info(
                    f"Using local API with model: {local_settings.get('name')} "
                    f"(SSL verification: {verify_ssl})"
                )
                
            elif llm_choice == "azure":
                # Load environment variables
                load_dotenv()
                os.environ["OPENAI_API_TYPE"] = "azure"
                
                # Use Azure settings
                azure_settings = model_settings.get('azure', {})
                
                # Handle temperature: use temperature if set, otherwise use average of temperature_low and temperature_high
                if 'temperature' in azure_settings:
                    temperature = float(azure_settings.get('temperature', 0.0))
                else:
                    temp_low = azure_settings.get('temperature_low', 0.0)
                    temp_high = azure_settings.get('temperature_high', 0.0)
                    temperature = (float(temp_low) + float(temp_high)) / 2.0
                
                self.llm = AzureChatOpenAI(
                    azure_deployment=azure_settings.get('name', 'gpt-4'),
                    temperature=temperature,
                    openai_api_version=os.getenv(
                        "OPENAI_API_VERSION",
                        "2024-05-01-preview"
                    ),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY_WUS")
                )
                self.logger.info(
                    f"Using Azure OpenAI API with model: {azure_settings.get('name')} "
                    f"(temperature: {temperature})"
                )
                
            else:  # openai
                # Load environment variables
                load_dotenv()
                
                # Use OpenAI settings
                openai_settings = model_settings.get('openai', {})
                model_name = openai_settings.get('name', 'gpt-3.5-turbo')
                
                # Handle temperature: use temperature if set, otherwise use average of temperature_low and temperature_high
                if 'temperature' in openai_settings:
                    temperature = float(openai_settings.get('temperature', 0.0))
                else:
                    temp_low = openai_settings.get('temperature_low', 0.0)
                    temp_high = openai_settings.get('temperature_high', 0.0)
                    temperature = (float(temp_low) + float(temp_high)) / 2.0
                
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=os.getenv("OPENAI_API_KEY_SS")
                )
                self.logger.info(
                    f"Using OpenAI API with model: {model_name} "
                    f"(temperature: {temperature})"
                )
                
        except Exception as e:
            self.logger.error(f"Error setting up LLMs: {e}")
            raise
        
    def _get_formatted_prompt(self) -> str:
        """Get formatted prompt for histology classification.

        Uses prompt from config if available, otherwise falls back to default.

        Returns:
            Formatted prompt string
        """
        if self.raw_prompt:
            # Use prompt from YAML config
            self.logger.debug("Using histology_classifier prompt from config")
            return self.raw_prompt
        
        # Fallback to default prompt if not in config
        self.logger.debug("Using default histology classifier prompt")
        return """You are an expert pathologist specialized in lung cancer classification.

MEDICAL REPORTS TO ANALYZE:
{reports}

EXAMPLE CASES:
1. Clear Single Type Cases:
Case 1: "Microscopic examination reveals acinar predominant adenocarcinoma (60%) with lepidic pattern (40%)"
"main_category": "epithelial_tumors", "specific_type": "Acinar adenocarcinoma"

Case 2: "Small cell lung cancer, WHO grade 3, extensive stage"
"main_category": "neuroendocrine_neoplasms", "specific_type": "Small cell carcinoma"

2. Mixed Pattern Cases:
Case 3: "Mixed adenocarcinoma with papillary (40%), acinar (35%), and solid (25%) patterns"
"main_category": "epithelial_tumors", "specific_type": "Papillary adenocarcinoma"

3. Unclear/Complex Cases:
Case 4: "Carcinoma with mixed small cell and squamous features"
"main_category": "neuroendocrine_neoplasms", "specific_type": "Small cell carcinoma"

CLASSIFICATION GUIDELINES:
1. Pattern Recognition Rules:
   - Look for specific histological patterns (acinar, lepidic, papillary, etc.)
   - Consider both descriptive terms and standardized medical terminology

2. Language Processing:
   - Check for terms in multiple languages (English, Korean, Latin medical terms)

Format your response exactly like this:
{{
    "main_category": "The main tumor category",
    "type": "The specific cancer type",
    "reasoning": "Detailed explanation citing report evidence",
    "confidence": "high/medium/low"
}}"""

    def _setup_agents(self):
        """Setup histology classification prompt template."""
        try:
            self.output_parser = HistologyOutputParser()
            
            # Format prompt with placeholders
            raw_prompt = self._get_formatted_prompt()
            
            # Prepare values for placeholders with token optimization
            from src.utils.data_utils import format_histology_structure_for_prompt
            
            # Use compact format for classification structure (reduces tokens)
            classification_structure = (
                format_histology_structure_for_prompt(self.histology_structure)
                if self.histology_structure
                else "No classification structure available"
            )
            
            # Format rules more compactly (extract only note text)
            import json
            if self.histology_rules:
                rules_text = []
                for rule in self.histology_rules:
                    note = rule.get('note', '')
                    if note:
                        rules_text.append(f"- {note}")
                classification_rules = "\n".join(rules_text) if rules_text else "No special classification rules"
            else:
                classification_rules = "No special classification rules"
            
            # Format prompt with all placeholders
            # Keep {reports} as placeholder for runtime replacement
            # Replace other placeholders with actual values
            try:
                # First try with all placeholders
                self.system_prompt_template = raw_prompt.format(
                    reports="{reports}",  # Keep as placeholder
                    classification_structure=classification_structure,
                    classification_rules=classification_rules
                )
            except KeyError as e:
                # If some placeholders are missing, try with just reports
                self.logger.debug(
                    f"Some placeholders not found in prompt: {e}. "
                    f"Using reports placeholder only."
                )
                try:
                    self.system_prompt_template = raw_prompt.format(reports="{reports}")
                except KeyError:
                    # If even reports placeholder is missing, use as-is
                    self.logger.warning(
                        "No {reports} placeholder found in prompt. "
                        "Using prompt as-is."
                    )
                    self.system_prompt_template = raw_prompt
            
        except Exception as e:
            self.logger.error(f"Failed to setup agents: {e}")
            raise

    def histology_classifier_node(self, state: Dict) -> Dict[str, Any]:
        """Process histology classification with better state handling.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with histology classification results
        """
        try:
            # Get available reports with priority
            input_data = self._prepare_input_data(state)
            
            if input_data["reports"] == "No medical reports available":
                return self._create_empty_result(
                    state,
                    "No medical reports available"
                )

            # Format prompt with actual reports
            formatted_prompt = self.system_prompt_template.format(
                reports=input_data["reports"]
            )
            
            # Create messages for LLM
            messages = [
                SystemMessage(content=formatted_prompt),
                HumanMessage(content=f"Medical Reports:\n{input_data['reports']}")
            ]

            # Get classification from LLM
            try:
                response = self.llm.invoke(messages)
                
                # Extract the output from the response
                if hasattr(response, 'content'):
                    output_text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    output_text = response['content']
                elif isinstance(response, dict) and 'output' in response:
                    output_text = response['output']
                else:
                    output_text = str(response)

                # Parse the output with improved error handling
                try:
                    parsed_result = self.output_parser.parse(output_text)
                    if isinstance(parsed_result, dict) and 'main_category' in parsed_result:
                        histology_result = {
                            "histology_category": parsed_result['main_category'],
                            "histology_type": parsed_result['type'],
                            "histology_confidence": parsed_result.get(
                                'confidence',
                                'low'
                            ),
                            "histology_reason": parsed_result.get('reasoning', ''),
                            "histology_raw": parsed_result.get('raw_text')
                        }
                    else:
                        return self._create_empty_result(
                            state,
                            "Invalid parser output format"
                        )
                except Exception as parse_error:
                    self.logger.error(
                        f"Error parsing output: {str(parse_error)}"
                    )
                    return self._create_empty_result(
                        state,
                        f"Parser error: {str(parse_error)}"
                    )
                
                self.logger.info(
                    f"Histology classification completed from "
                    f"{input_data['source_description']}: "
                    f"{histology_result['histology_category']} - "
                    f"{histology_result['histology_type']}"
                )
                
                return histology_result
                
            except Exception as e:
                self.logger.error(
                    f"Error in histology classification: {str(e)}"
                )
                return self._create_empty_result(state, str(e))
                
        except Exception as e:
            self.logger.error(f"Error in histology classifier node: {str(e)}")
            return self._create_empty_result(state, str(e))

    def _create_empty_result(
        self,
        state: Dict,
        error_message: str = "No data available"
    ) -> Dict:
        """Create empty result dictionary for error cases.

        Args:
            state: Current workflow state
            error_message: Error message to include

        Returns:
            Dictionary with empty/error histology classification
        """
        return {
            "histology_category": self.output_parser.NOT_FOUND_CATEGORY,
            "histology_type": self.output_parser.NOT_FOUND_TYPE,
            "histology_confidence": "low",
            "histology_reason": error_message,
            "histology_raw": None
        }
    
    def _prepare_input_data(self, state: Dict) -> Dict:
        """Prepare input data with report priority handling.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with reports and source description
        """
        report_data, sources = self._get_available_reports(state)
        
        if not report_data:
            return {
                "reports": "No medical reports available",
                "source_description": "No reports available"
            }
            
        self.logger.info(f"Using reports from: {sources}")
        
        # Log pathology report existence
        if "pathology" in sources:
            self.logger.info(
                "Using pathology report for histology classification"
            )
        else:
            self.logger.info(
                "No pathology report available, using alternative reports"
            )
        
        return {
            "reports": report_data,
            "source_description": ", ".join(sources)
        }

    def _get_available_reports(
        self,
        state: Dict
    ) -> Tuple[str, List[str]]:
        """Get available reports in priority order.

        Args:
            state: Current workflow state

        Returns:
            Tuple of (combined_report_text, list_of_sources)
        """
        # First check pathology report
        pathology_report = state['input'].get('pathology')
        if pathology_report:
            return pathology_report, ["pathology"]
            
        # If no pathology report, check other reports in priority order
        report_priority = [
            ('ebus', state['input'].get('ebus')),
            ('pet', state['input'].get('pet')),
            ('chest_ct', state['input'].get('chest_ct')),
            ('neck_biopsy', state['input'].get('neck_biopsy')),
            ('brain_mr', state['input'].get('brain_mr')),
            ('bone_scan', state['input'].get('bone_scan')),
            ('abdomen_pelvis_ct', state['input'].get('abdomen_pelvis_ct')),
            ('adrenal_ct', state['input'].get('adrenal_ct'))
        ]
        
        available_reports = []
        report_sources = []
        
        for source, report in report_priority:
            if report:
                available_reports.append(report)
                report_sources.append(source)
        
        combined_report = (
            "\n\n".join(available_reports)
            if available_reports
            else None
        )
        
        return combined_report, report_sources

