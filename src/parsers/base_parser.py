"""Base parser for TNM classifications."""

import json
import logging
import re
from typing import Any, ClassVar, Dict, Optional, TypedDict

logger = logging.getLogger(__name__)


class TNMClassificationDict(TypedDict):
    """TNM classification dictionary structure."""

    classification: str


class BaseTNMParser:
    """Base parser for TNM classifications with fixed dictionary handling."""

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def parse_base(self, text: str) -> Dict:
        """Base parsing logic for all TNM classifications.

        Args:
            text: Input text or dictionary to parse

        Returns:
            Dictionary with 'classification' key

        Raises:
            ValueError: If no valid classification found
        """
        try:
            self.logger.debug(f"\n{'='*50}\nParsing TNM classification")
            self.logger.debug(f"Input text (first 1000 chars): {str(text)[:1000]}")

            # Step 1: Handle dictionary input
            if isinstance(text, dict):
                self.logger.debug("Processing dictionary input")
                if 'classification' in text:
                    self.logger.debug(
                        f"Direct classification found: {text['classification']}"
                    )
                    return self._validate_data(text)

                if 'output' in text or 'content' in text:
                    parse_text = text.get('output', '') or text.get('content', '')
                    self.logger.debug(f"Found output/content: {parse_text}")
                    json_result = self._try_json_parsing(parse_text)
                    if json_result:
                        return self._validate_data(json_result)
                    text = str(parse_text).strip()
            else:
                text = str(text).strip()

            # Step 2: Try JSON code block parsing
            if json_result := self._try_json_parsing(text):
                self.logger.debug(f"JSON parsing result: {json_result}")
                return self._validate_data(json_result)

            # Step 3: Try pattern matching
            if extracted_result := self._try_extract(text):
                self.logger.debug(
                    f"Pattern matching result: {extracted_result}"
                )
                return self._validate_data(extracted_result)

            # Log the full text for debugging if all parsing attempts failed
            self.logger.warning(
                f"All parsing attempts failed. Full text (first 2000 chars): "
                f"{str(text)[:2000]}"
            )
            raise ValueError("Could not find valid classification in the output")

        except Exception as e:
            self.logger.error(
                f"Failed to parse TNM classification output: {str(e)}"
            )
            self.logger.debug(f"Problematic text (first 2000 chars): {str(text)[:2000]}")
            raise

    def _try_json_parsing(self, text: str) -> Optional[Dict]:
        """Try to find and parse JSON in the text using multiple approaches.

        Args:
            text: Text to parse for JSON

        Returns:
            Parsed JSON dictionary or None if not found
        """
        self.logger.debug(f"Attempting JSON parsing on text (first 500 chars): {text[:500]}")
        try:
            # Remove think tags first
            text = re.sub(
                r'<think>.*?</think>',
                '',
                text,
                flags=re.DOTALL
            )

            # Clean up text - remove trailing characters that might interfere
            text = re.sub(r'\}\s*"\s*\}*\s*$', '}', text)

            # Approach 1: Try original strict JSON block pattern
            strict_pattern = (
                r'```json\n\{\n\s*"classification":\s*"[^"]+"\n\}\n```'
            )
            match = re.search(strict_pattern, text)
            if match:
                json_text = match.group(0).replace('```json\n', '').replace(
                    '\n```', ''
                )
                try:
                    data = json.loads(json_text)
                    if 'classification' in data:
                        self.logger.info(
                            f"Found JSON using strict pattern: {data}"
                        )
                        return data
                except json.JSONDecodeError:
                    self.logger.debug(
                        "Strict pattern failed, trying next approach"
                    )

            # Approach 2: Try flexible JSON block pattern
            flexible_pattern = r'```json\s*(\{[^`]+?\})\s*```'
            matches = re.finditer(flexible_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json_text = match.group(1).strip()
                    data = json.loads(json_text)
                    if 'classification' in data:
                        self.logger.info(
                            f"Found JSON using flexible pattern: {data}"
                        )
                        return data
                except json.JSONDecodeError:
                    continue

            # Approach 3: Try to find any JSON object pattern
            json_object_pattern = r'\{\s*"classification"\s*:\s*"[^"]+"\s*\}'
            matches = re.finditer(json_object_pattern, text)
            for match in matches:
                try:
                    json_text = match.group(0)
                    data = json.loads(json_text)
                    if 'classification' in data:
                        self.logger.info(
                            f"Found JSON using object pattern: {data}"
                        )
                        return data
                except json.JSONDecodeError:
                    continue

            # Approach 4: Try direct value extraction
            direct_pattern = r'"classification"\s*:\s*"([^"]+)"'
            match = re.search(direct_pattern, text)
            if match:
                classification = match.group(1)
                data = {"classification": classification}
                self.logger.info(
                    f"Found classification using direct extraction: {data}"
                )
                return data

            # Approach 5: Additional fallback for bare JSON format
            fallback_matches = re.finditer(r'\{[^}]+\}', text)
            for match in fallback_matches:
                try:
                    json_text = match.group(0)
                    if '"classification"' in json_text:
                        data = json.loads(json_text)
                        if 'classification' in data:
                            self.logger.info(
                                f"Found JSON using fallback: {data}"
                            )
                            return data
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            self.logger.debug(f"JSON parsing failed: {e}")

        return None

    def _validate_data(self, data: Dict) -> TNMClassificationDict:
        """Validate and clean classification data.

        Args:
            data: Dictionary containing classification data

        Returns:
            Validated TNMClassificationDict

        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate data type
            if not isinstance(data, dict):
                raise ValueError(f"Invalid data type: {type(data)}")

            # Validate classification field type
            if not isinstance(data.get('classification'), str):
                raise ValueError("Classification must be a string")

            # Handle empty strings and whitespace
            classification = str(data['classification']).strip()
            if not classification:
                raise ValueError("Classification cannot be empty")

            return {'classification': classification}
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

    def _try_extract(self, text: str) -> Optional[Dict]:
        """Extract classification using appropriate function.

        Args:
            text: Text to extract classification from

        Returns:
            Dictionary with classification or None

        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        # This should be implemented by child classes
        raise NotImplementedError

