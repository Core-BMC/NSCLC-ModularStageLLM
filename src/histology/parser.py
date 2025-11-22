"""Histology output parser for extracting structured classification from LLM responses."""

import json
import logging
import re
from typing import ClassVar, Dict

from langchain_core.output_parsers import BaseOutputParser

try:
    from src.histology.classification import HistologyClassification
except ImportError:
    # Fallback for relative import
    from .classification import HistologyClassification


logger = logging.getLogger(__name__)


class HistologyOutputParser(BaseOutputParser[HistologyClassification]):
    """Parser for histology classification with simplified output structure."""

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)
    
    NOT_FOUND_CATEGORY: ClassVar[str] = "Not Determined"
    NOT_FOUND_TYPE: ClassVar[str] = "Insufficient Data"
    
    def get_format_instructions(self) -> str:
        """Get format instructions for LLM."""
        return """Your response must be formatted as a JSON object:
        {{
            "main_category": "The broad category (e.g., epithelial_tumors)",
            "type": "The specific cancer type",
            "reasoning": "Detailed explanation of classification based on report findings",
            "confidence": "high/medium/low"
        }}"""

    def parse(self, text: str) -> HistologyClassification:
        """Parse LLM response into structured histology classification.

        Args:
            text: Raw text response from LLM

        Returns:
            HistologyClassification dictionary
        """
        try:
            # Handle various input types
            if isinstance(text, dict):
                if 'output' in text:
                    text = text['output']
                elif 'content' in text:
                    text = text['content']
                elif isinstance(text, dict) and all(
                    key in text for key in [
                        'main_category', 'type', 'reasoning', 'confidence'
                    ]
                ):
                    # If the text is already a properly formatted dictionary,
                    # return it directly
                    return HistologyClassification(
                        main_category=text['main_category'],
                        type=text['type'],
                        reasoning=text['reasoning'],
                        confidence=text.get('confidence', 'medium'),
                        raw_text=None,
                        subcategory=text.get('subcategory', '')
                    )
            
            text = str(text).strip()
            
            # Log the full response for debugging
            logger.debug(f"Parsing histology response (first 1000 chars): {text[:1000]}")
            
            json_attempts = []
            
            # Approach 1: Try to find JSON code block first
            json_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            json_block_match = re.search(json_block_pattern, text, re.DOTALL)
            if json_block_match:
                try:
                    json_text = json_block_match.group(1).strip()
                    logger.debug(f"Found JSON code block: {json_text[:500]}")
                    data = json.loads(json_text)
                    if self._validate_fields(data):
                        logger.info(f"Valid histology classification found: {data.get('type')}")
                        return HistologyClassification(
                            main_category=data['main_category'],
                            type=data['type'],
                            reasoning=data['reasoning'],
                            confidence=data.get('confidence', 'medium'),
                            raw_text=None,
                            subcategory=data.get('subcategory', '')
                        )
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON block decode error: {e}")
                    json_attempts.append(f"JSON block decode error: {str(e)}")
            
            # Approach 2: Try to find complete JSON object (greedy match)
            # Find the first { and try to match until the last }
            json_start = text.find('{')
            if json_start != -1:
                # Try to find the matching closing brace
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    try:
                        json_text = text[json_start:json_end]
                        logger.debug(f"Extracted JSON object: {json_text[:500]}")
                        data = json.loads(json_text)
                        if self._validate_fields(data):
                            logger.info(f"Valid histology classification found: {data.get('type')}")
                            return HistologyClassification(
                                main_category=data['main_category'],
                                type=data['type'],
                                reasoning=data['reasoning'],
                                confidence=data.get('confidence', 'medium'),
                                raw_text=None,
                                subcategory=data.get('subcategory', '')
                            )
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON object decode error: {e}")
                        json_attempts.append(f"JSON object decode error: {str(e)}")
                        
                        # Try to fix truncated JSON by adding missing closing braces
                        if "Expecting" in str(e) or "Unterminated" in str(e):
                            try:
                                # Count open and close braces
                                open_braces = json_text.count('{')
                                close_braces = json_text.count('}')
                                missing_braces = open_braces - close_braces
                                
                                if missing_braces > 0:
                                    # Try to complete the JSON
                                    fixed_json = json_text + '}' * missing_braces
                                    logger.debug(f"Attempting to fix truncated JSON")
                                    data = json.loads(fixed_json)
                                    if self._validate_fields(data):
                                        logger.info(f"Fixed truncated JSON, found: {data.get('type')}")
                                        return HistologyClassification(
                                            main_category=data['main_category'],
                                            type=data['type'],
                                            reasoning=data.get('reasoning', '')[:500],  # Truncate if too long
                                            confidence=data.get('confidence', 'medium'),
                                            raw_text=None,
                                            subcategory=data.get('subcategory', '')
                                        )
                            except Exception as fix_error:
                                logger.debug(f"Failed to fix JSON: {fix_error}")
                                json_attempts.append(f"Fix attempt failed: {str(fix_error)}")
            
            # Approach 3: Try non-greedy pattern matching (fallback)
            json_pattern = r'\{[\s\S]*?\}'
            matches = re.finditer(json_pattern, text)
            for match in matches:
                try:
                    json_text = match.group()
                    logger.debug(f"Attempting to parse JSON: {json_text[:500]}")
                    data = json.loads(json_text)
                    if self._validate_fields(data):
                        logger.info(f"Valid histology classification found: {data.get('type')}")
                        return HistologyClassification(
                            main_category=data['main_category'],
                            type=data['type'],
                            reasoning=data['reasoning'],
                            confidence=data.get('confidence', 'medium'),
                            raw_text=None,
                            subcategory=data.get('subcategory', '')
                        )
                    else:
                        logger.warning(f"JSON parsed but validation failed: {data}")
                        json_attempts.append(f"Validation failed: {data}")
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error: {e}, text: {match.group()[:200]}")
                    json_attempts.append(f"JSON decode error: {str(e)}")
                    continue
            
            error_msg = "Could not parse valid JSON from response"
            if json_attempts:
                error_msg += f". Attempts: {json_attempts[:3]}"  # Show first 3 attempts
            logger.warning(f"{error_msg}. Full response: {text[:500]}")
            return self._create_empty_result(error_msg)
            
        except Exception as e:
            logger.error(f"Error parsing histology classification: {e}")
            return self._create_empty_result(str(e))

    def _validate_fields(self, data: dict) -> bool:
        """Validate required fields are present and non-empty.

        Args:
            data: Dictionary to validate

        Returns:
            True if all required fields are present and valid
        """
        required_fields = ['main_category', 'type', 'reasoning']
        return all(
            field in data and data[field] and isinstance(data[field], str)
            for field in required_fields
        )

    def _create_empty_result(
        self,
        error_message: str = "No data available"
    ) -> HistologyClassification:
        """Create empty result dictionary for error cases.

        Args:
            error_message: Error message to include in result

        Returns:
            Empty HistologyClassification with error information
        """
        empty_result = HistologyClassification(
            main_category=self.NOT_FOUND_CATEGORY,
            type=self.NOT_FOUND_TYPE,
            reasoning=error_message,
            confidence="low",
            raw_text=None,
            subcategory=""
        )
        self.logger.warning(f"Creating empty result: {error_message}")
        return empty_result

