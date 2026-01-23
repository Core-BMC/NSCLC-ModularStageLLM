"""TNM classification parsers for T, N, M categories."""

import json
import logging
import re
from typing import Any, ClassVar, Dict, Optional, Set

from langchain_core.output_parsers import BaseOutputParser

from src.parsers.base_parser import BaseTNMParser, TNMClassificationDict
from src.utils.tnm_extraction_utils import (
    extract_m_classification,
    extract_n_classification,
    extract_t_classification,
    normalize_n_classification,
    normalize_t_classification,
)

logger = logging.getLogger(__name__)


class TNM_T_Parser(BaseTNMParser):
    """Parser for T category classification."""

    VALID_T_CLASSIFICATIONS: ClassVar[Set[str]] = {
        'T0', 'Tis', 'T1mi', 'T1a', 'T1b', 'T1c', 'T2a', 'T2b',
        'T3', 'T4', 'T1', 'T2', 'Tx'
    }

    def normalize_classification(self, t_val: str) -> str:
        """Normalize T classification to standardized format.

        Args:
            t_val: T classification value to normalize

        Returns:
            Normalized T classification string
        """
        self.logger.debug(f"T Parser - Normalizing value: {t_val}")

        # Use utility function for normalization
        normalized = normalize_t_classification(t_val)
        self.logger.debug(f"T Parser - Mapping result: {normalized}")

        # Check if normalized value is valid
        if normalized in self.VALID_T_CLASSIFICATIONS:
            return normalized

        # Fallback: check if original value is valid
        if t_val in self.VALID_T_CLASSIFICATIONS:
            return t_val

        return normalized

    def _validate_data(self, data: dict) -> TNMClassificationDict:
        """Validate T classification specific data.

        Args:
            data: Dictionary containing T classification data

        Returns:
            Validated TNMClassificationDict with normalized T classification

        Raises:
            ValueError: If classification is invalid
        """
        self.logger.debug(f"T Parser - Input data: {data}")
        clean_data = super()._validate_data(data)
        classification = clean_data['classification']
        self.logger.debug(f"T Parser - Clean classification: {classification}")

        # Normalize classification
        normalized_classification = self.normalize_classification(classification)
        self.logger.debug(
            f"T Parser - Normalized classification: {normalized_classification}"
        )

        if normalized_classification.upper() not in {
            c.upper() for c in self.VALID_T_CLASSIFICATIONS
        }:
            raise ValueError(
                f"Invalid T classification: {normalized_classification}"
            )

        return {'classification': normalized_classification}

    def _try_extract(self, text: str) -> Optional[Dict]:
        """Extract T classification using pattern matching.

        Args:
            text: Text to extract T classification from

        Returns:
            Dictionary with T classification or None
        """
        result = extract_t_classification(text)
        if result:
            result['classification'] = self.normalize_classification(
                result['classification']
            )
            return result
        return None


class TNM_N_Parser(BaseTNMParser):
    """Parser for N category classification."""

    VALID_N_CLASSIFICATIONS: ClassVar[Set[str]] = {
        'N0', 'N1', 'N2', 'N3', 'Nx', 'N1a', 'N1b', 'N2a', 'N2b'
    }

    def normalize_classification(self, n_val: str) -> str:
        """Normalize N classification to standardized format.

        Args:
            n_val: N classification value to normalize

        Returns:
            Normalized N classification string
        """
        # Check if already a valid classification (preserve exact format)
        if n_val in self.VALID_N_CLASSIFICATIONS:
            return n_val
        
        # Normalize case and check if it matches a valid classification
        n_upper = n_val.upper()
        format_map = {
            'N0': 'N0',
            'N1': 'N1',
            'N1A': 'N1a' if 'N1a' in self.VALID_N_CLASSIFICATIONS else 'N1',
            'N1B': 'N1b' if 'N1b' in self.VALID_N_CLASSIFICATIONS else 'N1',
            'N2': 'N2',
            'N2A': 'N2a' if 'N2a' in self.VALID_N_CLASSIFICATIONS else 'N2',
            'N2B': 'N2b' if 'N2b' in self.VALID_N_CLASSIFICATIONS else 'N2',
            'N3': 'N3',
            'NX': 'N0',
        }
        
        # Use format_map if available
        if n_upper in format_map:
            normalized = format_map[n_upper]
            # Only return if it's valid
            if normalized in self.VALID_N_CLASSIFICATIONS:
                return normalized
        
        # Fallback to utility function
        normalized = normalize_n_classification(n_val)
        # If normalized value is valid, return it
        if normalized in self.VALID_N_CLASSIFICATIONS:
            return normalized
        
        return n_val

    def _try_extract(self, text: str) -> Optional[Dict]:
        """Extract N classification using pattern matching.

        Args:
            text: Text to extract N classification from

        Returns:
            Dictionary with N classification or None
        """
        result = extract_n_classification(text)
        if result:
            result['classification'] = self.normalize_classification(
                result['classification']
            )
            return result
        return None

    def _validate_data(self, data: dict) -> TNMClassificationDict:
        """Validate N classification specific data.

        Args:
            data: Dictionary containing N classification data

        Returns:
            Validated TNMClassificationDict with normalized N classification

        Raises:
            ValueError: If classification is invalid
        """
        clean_data = super()._validate_data(data)
        classification = clean_data['classification']

        # Normalize classification
        normalized_classification = self.normalize_classification(classification)

        if normalized_classification not in self.VALID_N_CLASSIFICATIONS:
            raise ValueError(
                f"Invalid N classification: {normalized_classification}"
            )

        return {'classification': normalized_classification}


class TNM_M_Parser(BaseTNMParser):
    """Parser for M category classification."""

    VALID_M_CLASSIFICATIONS: ClassVar[Set[str]] = {
        'M0', 'M1a', 'M1b', 'M1c', 'M1c1', 'M1c2', 'Mx', 'M1'
    }

    def normalize_classification(self, m_val: str) -> str:
        """Normalize M classification to standardized format.

        Args:
            m_val: M classification value to normalize

        Returns:
            Normalized M classification string
        """
        m_upper = m_val.upper()
        format_map = {
            'M0': 'M0',
            'M1A': 'M1a',
            'M1B': 'M1b',
            'M1C': 'M1c',
            'M1C1': 'M1c1',
            'M1C2': 'M1c2',
            'MX': 'M0',
            'Mx': 'M0',
            'M1': 'M1a',
        }
        # Check if already a valid classification (preserve exact format)
        if m_val in self.VALID_M_CLASSIFICATIONS:
            return m_val
        # Use format_map for normalization
        normalized = format_map.get(m_upper)
        if normalized:
            return normalized
        # If not in format_map but uppercase matches a valid classification, normalize case
        if m_upper in {c.upper() for c in self.VALID_M_CLASSIFICATIONS}:
            # Find matching valid classification and return it
            for valid in self.VALID_M_CLASSIFICATIONS:
                if valid.upper() == m_upper:
                    return valid
        return m_val

    def _try_extract(self, text: str) -> Optional[Dict]:
        """Extract M classification using pattern matching.

        Args:
            text: Text to extract M classification from

        Returns:
            Dictionary with M classification or None
        """
        result = extract_m_classification(text)
        if result:
            result['classification'] = self.normalize_classification(
                result['classification']
            )
            return result
        return None

    def _validate_data(self, data: dict) -> TNMClassificationDict:
        """Validate M classification specific data.

        Args:
            data: Dictionary containing M classification data

        Returns:
            Validated TNMClassificationDict with normalized M classification

        Raises:
            ValueError: If classification is invalid
        """
        clean_data = super()._validate_data(data)
        classification = clean_data['classification']

        # Normalize classification
        normalized_classification = self.normalize_classification(classification)

        if normalized_classification not in self.VALID_M_CLASSIFICATIONS:
            raise ValueError(
                f"Invalid M classification: {normalized_classification}"
            )

        return {'classification': normalized_classification}


class TNMOutputParser(BaseOutputParser[TNMClassificationDict]):
    """Unified parser for TNM classification output."""

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    VALID_T_CLASSIFICATIONS: ClassVar[Set[str]] = {
        'T0', 'Tis', 'T1mi', 'T1a', 'T1b', 'T1c', 'T2a', 'T2b',
        'T3', 'T4', 'T1', 'T2', 'Tx'
    }
    VALID_N_CLASSIFICATIONS: ClassVar[Set[str]] = {
        'N0', 'N1', 'N2', 'N3', 'Nx', 'N2a', 'N2b'
    }
    VALID_M_CLASSIFICATIONS: ClassVar[Set[str]] = {
        'M0', 'M1a', 'M1b', 'M1c', 'Mx', 'M1', 'M1c1', 'M1c2'
    }

    def normalize_t_classification(self, t_val: str) -> str:
        """Normalize T classification to standardized format.

        Args:
            t_val: T classification value to normalize

        Returns:
            Normalized T classification string
        """
        t_upper = t_val.upper()
        self.logger.debug(
            f"T Parser - Normalizing value: {t_val} (upper: {t_upper})"
        )

        format_map = {
            'T0': 'T0',
            'TX': 'T0',
            'Tx': 'T0',
            'T1': 'T1a',
            'T2': 'T2a',
            'TIS': 'T1a',
            'T1MI': 'T1a',
            'Tis': 'T1a',
            'T1mi': 'T1a'
        }

        normalized = format_map.get(t_upper)
        self.logger.debug(f"T Parser - Mapping result: {normalized}")

        if normalized:
            return normalized

        # Check if original value is a valid classification
        if t_val in self.VALID_T_CLASSIFICATIONS:
            return t_val

        return t_val

    def normalize_n_classification(self, n_val: str) -> str:
        """Normalize N classification to standardized format.

        Args:
            n_val: N classification value to normalize

        Returns:
            Normalized N classification string
        """
        if n_val in self.VALID_N_CLASSIFICATIONS:
            return n_val

        n_upper = n_val.upper()
        format_map = {
            'N0': 'N0',
            'N1': 'N1',
            'N1A': 'N1a' if 'N1a' in self.VALID_N_CLASSIFICATIONS else 'N1',
            'N1B': 'N1b' if 'N1b' in self.VALID_N_CLASSIFICATIONS else 'N1',
            'N2': 'N2',
            'N2A': 'N2a' if 'N2a' in self.VALID_N_CLASSIFICATIONS else 'N2',
            'N2B': 'N2b' if 'N2b' in self.VALID_N_CLASSIFICATIONS else 'N2',
            'N3': 'N3',
            'NX': 'N0'
        }
        return format_map.get(n_upper, n_val)

    def normalize_m_classification(self, m_val: str) -> str:
        """Normalize M classification to standardized format.

        Args:
            m_val: M classification value to normalize

        Returns:
            Normalized M classification string
        """
        m_upper = m_val.upper()
        format_map = {
            'M0': 'M0',
            'M1A': 'M1a',
            'M1B': 'M1b',
            'M1C': 'M1c',
            'M1C1': 'M1c1',
            'M1C2': 'M1c2',
            'MX': 'M0',
            'Mx': 'M0',
            'M1': 'M1a',
        }
        # Preserve M1c1 and M1c2 format (M1c1, M1c2)
        if m_upper in ['M1C1', 'M1C2']:
            return format_map[m_upper]
        return format_map.get(m_upper, m_val)

    def get_format_instructions(self) -> str:
        """Get format instructions for LLM output.

        Returns:
            Format instruction string
        """
        return """IMPORTANT: Analyze the provided report and determine TNM staging. Return ONLY a JSON object without any additional text or markdown formatting.

    You MUST choose EXACTLY ONE stage for each of T, N, and M classifications.

    Format your response as JSON:
    {{
      "t_classification": "CHOOSE ONE: T0/T1a/T1b/T1c/T2a/T2b/T3/T4",
      "n_classification": "CHOOSE ONE: N0/N1/N2/N3",
      "m_classification": "CHOOSE ONE: M0/M1a/M1b/M1c"
    }}

    Note:
    - Return only the JSON object, no explanations or additional text
    - Each classification must have exactly one value
    - Values must match the exact format shown above"""

    def parse(self, text: str) -> TNMClassificationDict:
        """Parse TNM classification from text.

        Args:
            text: Text containing TNM classification

        Returns:
            TNMClassificationDict with parsed classifications

        Raises:
            ValueError: If parsing fails
        """
        try:
            self.logger.debug(f"Attempting to parse text: {text}")

            if isinstance(text, dict):
                if 'output' in text:
                    text = text['output']
                elif all(key in text for key in [
                    't_classification',
                    'n_classification',
                    'm_classification'
                ]):
                    return self._validate_data(text)

            text = str(text).strip()

            # JSON parsing
            try:
                text = re.sub(r'```json\s*', '', text)
                text = re.sub(r'\s*```', '', text)
                pattern = r'{[^{}]*}'
                match = re.search(pattern, text)
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)
                    return self._validate_data(data)
            except Exception:
                self.logger.debug(
                    "JSON parsing failed, trying alternative methods"
                )

            # Simple pattern format
            try:
                clean_text = ' '.join(text.split())
                simple_patterns = [
                    r'[-\s]T(\d[abc]?|is|mi|0)\b',
                    r'[-\s]N([0-3])\b',
                    r'[-\s]M([0-1][abc]?)\b'
                ]

                matches = []
                for pattern in simple_patterns:
                    match = re.search(pattern, clean_text)
                    if match:
                        matches.append(match.group(1))

                if len(matches) == 3:
                    data = {
                        't_classification': f'T{matches[0]}',
                        'n_classification': f'N{matches[1]}',
                        'm_classification': f'M{matches[2]}'
                    }
                    return self._validate_data(data)
            except Exception as e:
                self.logger.debug(f"Simple format parsing failed: {str(e)}")

            # Method 2: Try to find dash format
            try:
                # Look for patterns like "- T3" or "• T3" or similar
                t_dash = r'[-•\*]\s*(?:T:?\s*)?([T]?[0-4](?:[abc]|is|mi)?)'
                n_dash = r'[-•\*]\s*(?:N:?\s*)?([N]?[0-3])'
                m_dash = r'[-•\*]\s*(?:M:?\s*)?([M]?[0-1](?:[abc])?)'

                t_match = re.search(t_dash, text, re.IGNORECASE)
                n_match = re.search(n_dash, text, re.IGNORECASE)
                m_match = re.search(m_dash, text, re.IGNORECASE)

                if t_match and n_match and m_match:
                    # Extract matches and ensure proper T, N, M prefixes
                    t_val = t_match.group(1)
                    n_val = n_match.group(1)
                    m_val = m_match.group(1)

                    # Add T, N, M prefixes if missing
                    if not t_val.upper().startswith('T'):
                        t_val = 'T' + t_val
                    if not n_val.upper().startswith('N'):
                        n_val = 'N' + n_val
                    if not m_val.upper().startswith('M'):
                        m_val = 'M' + m_val

                    data = {
                        't_classification': t_val.upper(),
                        'n_classification': n_val.upper(),
                        'm_classification': m_val.upper()
                    }
                    return self._validate_data(data)
            except Exception as e:
                self.logger.debug(
                    f"Dash/bullet point format parsing failed: {str(e)}"
                )

            # Method 3: Try to find True T/N/M in medical report format
            try:
                # Pattern for medical report format with multiple variations
                t_patterns = [
                    r'\* True T:\s*([T][0-4](?:[abc]|is|mi)?)',
                    r'True T:\s*([T][0-4](?:[abc]|is|mi)?)',
                    r'True T stage:\s*([T][0-4](?:[abc]|is|mi)?)',
                    r'T:\s*([T][0-4](?:[abc]|is|mi)?)\s*[(\-]',
                    r'T\s+([T][0-4](?:[abc]|is|mi)?)\s*[^\w]',
                ]

                n_patterns = [
                    r'\* True N:\s*([N][0-3])',
                    r'True N:\s*([N][0-3])',
                    r'True N stage:\s*([N][0-3])',
                    r'N:\s*([N][0-3])\s*[(\-]',
                    r'N\s+([N][0-3])\s*[^\w]',
                ]

                m_patterns = [
                    r'\* True M:\s*([M][0-1](?:[abc])?)',
                    r'True M:\s*([M][0-1](?:[abc])?)',
                    r'True M stage:\s*([M][0-1](?:[abc])?)',
                    r'M:\s*([M][0-1](?:[abc])?)\s*[(\-]',
                    r'M\s+([M][0-1](?:[abc])?)\s*[^\w]',
                ]

                t_class = None
                n_class = None
                m_class = None

                # Try each pattern until we find a match
                for pattern in t_patterns:
                    match = re.search(pattern, text)
                    if match:
                        t_class = match.group(1)
                        break

                for pattern in n_patterns:
                    match = re.search(pattern, text)
                    if match:
                        n_class = match.group(1)
                        break

                for pattern in m_patterns:
                    match = re.search(pattern, text)
                    if match:
                        m_class = match.group(1)
                        break

                if t_class and n_class and m_class:
                    data = {
                        't_classification': t_class,
                        'n_classification': n_class,
                        'm_classification': m_class
                    }
                    return self._validate_data(data)

            except Exception as e:
                self.logger.debug(
                    f"Medical report pattern matching failed: {str(e)}"
                )

            # Method 4: Try to find TNM combined pattern
            try:
                tnm_pattern = (
                    r'(?:^|\s)([T][0-4](?:[abc]|is|mi)?)([N][0-3])'
                    r'(?:M[x0-1](?:[abc])?)'
                )
                tnm_match = re.search(tnm_pattern, text)
                if tnm_match:
                    t_class = tnm_match.group(1)
                    n_class = tnm_match.group(2)

                    # Try to find M classification separately if it exists
                    m_matches = re.findall(r'[M][0-1](?:[abc])?', text)
                    m_class = (
                        m_matches[0] if m_matches else 'M0'
                    )  # Default to M0 if not found

                    data = {
                        't_classification': t_class,
                        'n_classification': n_class,
                        'm_classification': m_class
                    }
                    return self._validate_data(data)
            except Exception as e:
                self.logger.debug(
                    f"Combined TNM pattern matching failed: {str(e)}"
                )

            # Method 5: Try to find any TNM-like patterns with context
            try:
                # Context-aware patterns
                t_context = (
                    r'(?:tumor size|tumor|T classification|T stage|size|\bT\b)'
                    r'\D*?([T]?[0-4](?:[abc]|is|mi)?)'
                )
                n_context = (
                    r'(?:node|lymph|N classification|N stage|\bN\b)'
                    r'\D*?([N]?[0-3])'
                )
                m_context = (
                    r'(?:metastasis|metastases|M classification|M stage|\bM\b)'
                    r'\D*?([M]?[0-1](?:[abc])?)'
                )

                t_match = re.search(t_context, text, re.IGNORECASE)
                n_match = re.search(n_context, text, re.IGNORECASE)
                m_match = re.search(m_context, text, re.IGNORECASE)

                if t_match and n_match and m_match:
                    # Extract matches and ensure proper T, N, M prefixes
                    t_val = t_match.group(1)
                    n_val = n_match.group(1)
                    m_val = m_match.group(1)

                    # Add T, N, M prefixes if missing
                    if not t_val.upper().startswith('T'):
                        t_val = 'T' + t_val
                    if not n_val.upper().startswith('N'):
                        n_val = 'N' + n_val
                    if not m_val.upper().startswith('M'):
                        m_val = 'M' + m_val

                    data = {
                        't_classification': t_val.upper(),
                        'n_classification': n_val.upper(),
                        'm_classification': m_val.upper()
                    }
                    return self._validate_data(data)

            except Exception as e:
                self.logger.debug(f"Context pattern matching failed: {str(e)}")

            raise ValueError("Could not parse TNM classification output")

        except Exception as e:
            self.logger.error(
                f"Failed to parse TNM classification output: {str(e)}"
            )
            raise

    def _validate_data(self, data: dict) -> TNMClassificationDict:
        """Validate and normalize the parsed data.

        Args:
            data: Dictionary with TNM classifications

        Returns:
            Validated TNMClassificationDict

        Raises:
            ValueError: If validation fails
        """
        required_fields = [
            't_classification',
            'n_classification',
            'm_classification'
        ]

        # Check all required fields are present
        if not all(field in data for field in required_fields):
            missing = [
                field for field in required_fields if field not in data
            ]
            raise ValueError(f"Missing required fields: {missing}")

        # Normalize classifications
        t_class = self.normalize_t_classification(data['t_classification'])
        n_class = self.normalize_n_classification(data['n_classification'])
        m_class = self.normalize_m_classification(data['m_classification'])

        # Validate normalized classifications
        if t_class not in self.VALID_T_CLASSIFICATIONS:
            raise ValueError(f"Invalid T classification: {t_class}")

        if n_class not in self.VALID_N_CLASSIFICATIONS:
            raise ValueError(f"Invalid N classification: {n_class}")

        if m_class not in self.VALID_M_CLASSIFICATIONS:
            raise ValueError(f"Invalid M classification: {m_class}")

        return TNMClassificationDict(
            t_classification=t_class,
            n_classification=n_class,
            m_classification=m_class
        )
