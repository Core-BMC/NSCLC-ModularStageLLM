"""TNM classification extraction utilities.

This module provides functions for extracting T, N, and M classifications
from medical report text using pattern matching.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Exclusion patterns for filtering uncertain diagnoses
EXCLUSION_PATTERNS = [
    r'r/o',
    r'rule\s*out',
    r'versus',
    r'vs\.',
    r'suspicious\s*for',
    r'suspected',
    r'possible',
    r'probable',
    r'consider',
    r'suggestive\s*of',
    r'cannot\s*exclude',
    r'cannot\s*rule\s*out',
    r'differential\s*diagnosis',
    r'd/dx',
    r'ddx'
]

# MRI-related patterns to exclude
MRI_PATTERNS = [
    r'T[12]\s*(?:weighted|enhancement|signal|sequence|hyperintense|hypointense)',
    r'(?:in|on)\s+T[12]',
    r'T[12][WW]\s',
    r'T[12]\s+images?',
    r'(?:hyper|hypo)intense\s+(?:on|in)?\s*T[12]',
    r'T[12]\s*(?:-|\s+)(?:hyper|hypo)',
]


def has_exclusion_pattern(text_segment: str) -> bool:
    """Check if the text segment contains any exclusion patterns.

    Args:
        text_segment: Text segment to check

    Returns:
        True if exclusion pattern found, False otherwise
    """
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, text_segment, re.IGNORECASE):
            return True
    return False


def normalize_t_classification(t_val: str) -> str:
    """Normalize T classification to standardized format.

    Args:
        t_val: T classification value to normalize

    Returns:
        Normalized T classification string
    """
    t_upper = t_val.upper()
    format_map = {
        'T0': 'T0',
        'TIS': 'Tis',
        'T1MI': 'T1mi',
        'T1A': 'T1a',
        'T1B': 'T1b',
        'T1C': 'T1c',
        'T2A': 'T2a',
        'T2B': 'T2b',
        'T3': 'T3',
        'T4': 'T4'
    }
    return format_map.get(t_upper, t_val)


def normalize_n_classification(n_val: str) -> str:
    """Normalize N classification to standardized format.

    Args:
        n_val: N classification value to normalize

    Returns:
        Normalized N classification string
    """
    n_upper = n_val.upper()
    format_map = {
        'N0': 'N0',
        'N1': 'N1',
        'N1A': 'N1',
        'N1B': 'N1',
        'N2': 'N2',
        'N2A': 'N2',
        'N2B': 'N2',
        'N3': 'N3',
        'NX': 'N0',  # Convert NX to N0 as per requirement
        'N/A': 'N0'  # Handle N/A cases
    }
    return format_map.get(n_upper, n_val)


def determine_t_stage_by_size(size_cm: float) -> Dict[str, str]:
    """Determine T stage based on tumor size in centimeters.

    Args:
        size_cm: Tumor size in centimeters

    Returns:
        Dictionary with 'classification' key containing T stage
    """
    if size_cm <= 1:
        return {'classification': 'T1a'}  # ≤1cm
    elif size_cm <= 2:
        return {'classification': 'T1b'}  # >1cm but ≤2cm
    elif size_cm <= 3:
        return {'classification': 'T1c'}  # >2cm but ≤3cm
    elif size_cm <= 4:
        return {'classification': 'T2a'}  # >3cm but ≤4cm
    elif size_cm <= 5:
        return {'classification': 'T2b'}  # >4cm but ≤5cm
    elif size_cm <= 7:
        return {'classification': 'T3'}   # >5cm but ≤7cm
    else:
        return {'classification': 'T4'}   # >7cm


def extract_t_classification(text: str) -> Optional[Dict[str, Any]]:
    """Extract T classification from text with pattern matching.

    Args:
        text: Medical report text to analyze

    Returns:
        Dictionary with 'classification' key containing T stage,
        or None if no valid classification found
    """
    text = str(text).strip().lower()

    # Remove MRI-related patterns
    temp_text = text
    for pattern in MRI_PATTERNS:
        temp_text = re.sub(pattern, '', temp_text, flags=re.IGNORECASE)

    # Direct T stage patterns
    t_patterns = [
        r'(?:TNM|clinical|pathologic(?:al)?)\s*(?:staging|classification)?\s*[:\s]\s*T([0-4](?:[abc]|is|mi)?)',
        r'T\s*stage\s*[:\s]\s*([0-4](?:[abc]|is|mi)?)',
        r'(?:tumor|tumour|cancer)\s*stage\s*[:\s]\s*T([0-4](?:[abc]|is|mi)?)',
        r'\bcT([0-4](?:[abc]|is|mi)?)\b',
        r'\bpT([0-4](?:[abc]|is|mi)?)\b',
        r'R/O.*?(?:T([0-4](?:[abc]|is|mi)?)[N][0-3][M][0-1x])',
        r'R/O.*?T([0-4](?:[abc]|is|mi)?)\b',
        r'T stage:?\s*([T]?[0-4](?:[abc]|is|mi)?)',
        r'classification:?\s*([T]?[0-4](?:[abc]|is|mi)?)',
        r'staging:?\s*([T]?[0-4](?:[abc]|is|mi)?)',
        r'\bT\s*([0-4](?:[abc]|is|mi)?)\b'
    ]

    for pattern in t_patterns:
        match = re.search(pattern, temp_text, re.IGNORECASE)
        if match:
            # Check context for exclusion patterns
            start = max(0, match.start() - 20)
            end = min(len(temp_text), match.end() + 20)
            context = temp_text[start:end]

            if has_exclusion_pattern(context):
                continue

            t_val = match.group(1)
            if not t_val.upper().startswith('T'):
                t_val = 'T' + t_val
            return {'classification': normalize_t_classification(t_val)}

    # Size patterns in mm
    mm_patterns = [
        r'(?:nodule|nodular lesion|mass|tumor|lesion)(?:[^.]*?)(?:measuring|sized?|about|approximately|~)?\s*'
        r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)',
        r'(?:nodule|nodular lesion|mass|tumor|lesion)(?:[^.]*?)\((?:[^)]*?)(\d+(?:\.\d+)?)\s*mm',
        r'#[^,]*?(?:,\s*)?(\d+(?:\.\d+)?)\s*mm',
    ]

    for pattern in mm_patterns:
        match = re.search(pattern, temp_text, flags=re.IGNORECASE)
        if match:
            try:
                size_mm = float(match.group(1))
                size_cm = size_mm / 10  # Convert mm to cm
                return determine_t_stage_by_size(size_cm)
            except ValueError:
                continue

    # Invasion patterns
    invasion_patterns = [
        r'(?:invad(?:e|ing|es)|invasion of|extend(?:s|ing)? into)\s*'
        r'(?:chest wall|mediastinum|diaphragm|heart|vessels|trachea|nerve)',
        r'(?:pleural|pericardial)\s*(?:invasion|involvement|effusion)',
        r'involvement of\s*(?:adjacent|surrounding)\s*structures',
        r'direct\s*extension\s*into'
    ]

    for pattern in invasion_patterns:
        match = re.search(pattern, temp_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 20)
            end = min(len(temp_text), match.end() + 20)
            context = temp_text[start:end]

            if has_exclusion_pattern(context):
                continue

            return {'classification': 'T4'}

    # Location patterns
    location_patterns = [
        r'(?:main bronchus|carina|trachea)',
        r'(?:visceral pleura|parietal pleura|chest wall)'
    ]

    for pattern in location_patterns:
        match = re.search(pattern, temp_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 20)
            end = min(len(temp_text), match.end() + 20)
            context = temp_text[start:end]

            if has_exclusion_pattern(context):
                continue

            if re.search(r'(?:invad|invasion|extend)', context, re.IGNORECASE):
                if not has_exclusion_pattern(context):
                    return {'classification': 'T4'}
            else:
                if not has_exclusion_pattern(context):
                    return {'classification': 'T3'}

    return None


def extract_n_classification(text: str) -> Optional[Dict[str, Any]]:
    """Extract N classification from text with pattern matching.

    Args:
        text: Medical report text to analyze

    Returns:
        Dictionary with 'classification' key containing N stage,
        or None if no valid classification found (defaults to N0)
    """
    try:
        text = str(text).strip()

        # Direct N stage patterns
        n_patterns = [
            r'N stage:?\s*([N]?[0-3X])',
            r'classification:?\s*([N]?[0-3X])',
            r'staging:?\s*([N]?[0-3X])',
            r'\bN\s*([0-3X])\b',
            r'(?:T[0-4])?N([0-3X])(?:M[0-1X])?',  # TNM combined pattern
            r'cN([0-3X])',  # Clinical N stage
            r'pN([0-3X])'   # Pathological N stage
        ]

        for pattern in n_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                if not has_exclusion_pattern(context):
                    n_val = match.group(1)
                    if not n_val.upper().startswith('N'):
                        n_val = 'N' + n_val
                    return {'classification': normalize_n_classification(n_val)}

        # Lymph node description patterns
        lymph_patterns = [
            (r'no.*?(?:evidence|sign).*?(?:lymph\s*node.*?involvement|nodal\s*metastasis)', 'N0'),
            (r'no.*?(?:enlarged|suspicious|pathologic).*?lymph\s*node', 'N0'),
            (r'lymph\s*nodes.*?(?:normal|unremarkable|negative)', 'N0'),
            (r'(?:ipsilateral|same\s*side).*?(?:peribronchial|hilar).*?lymph\s*node.*?(?:involvement|metastasis)', 'N1'),
            (r'hilar.*?lymph\s*node.*?(?:involvement|metastasis)', 'N1'),
            (r'(?:ipsilateral|same\s*side).*?(?:mediastinal|subcarinal).*?lymph\s*node.*?(?:involvement|metastasis)', 'N2'),
            (r'mediastinal.*?lymph\s*node.*?(?:involvement|metastasis)', 'N2'),
            (r'(?:contralateral|opposite\s*side).*?lymph\s*node.*?(?:involvement|metastasis)', 'N3'),
            (r'supraclavicular.*?lymph\s*node.*?(?:involvement|metastasis)', 'N3'),
            (r'scalene.*?lymph\s*node.*?(?:involvement|metastasis)', 'N3')
        ]

        for pattern, n_class in lymph_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                context_start = max(0, match.start() - 20)
                context_end = min(len(text), match.end() + 20)
                context = text[context_start:context_end]
                if not has_exclusion_pattern(context):
                    return {'classification': n_class}

        # Default to N0 if no clear evidence
        return {'classification': 'N0'}

    except Exception as e:
        logger.error(f"Error in N classification extraction: {str(e)}")
        return {'classification': 'N0'}  # Safe default


def extract_m_classification(text: str) -> Optional[Dict[str, Any]]:
    """Extract M classification from text with pattern matching.

    Args:
        text: Medical report text to analyze

    Returns:
        Dictionary with 'classification' key containing M stage
    """
    text = str(text).strip()

    # M0 patterns: no metastasis
    m0_patterns = [
        r'no evidence of (?:metastasis|metastases|malignancy)',
        r'negative for (?:metastasis|metastases|malignancy)',
        r'without evidence of (?:metastasis|metastases)',
        r'no.*?(?:distant|remote).*?(?:metastasis|metastases)',
        r'brain.*?(?:no evidence|negative|without).*?(?:metastasis|metastases)',
        r'bone.*?(?:no evidence|negative|without).*?(?:metastasis|metastases)',
        r'liver.*?(?:no evidence|negative|without).*?(?:metastasis|metastases)',
        r'adrenal.*?(?:benign|hyperplasia|no evidence|negative)',
        r'benign (?:lesions?|changes?|hyperplasia)',
        r'physiologic uptake',
        r'no significant (?:lesions?|abnormalit(?:y|ies))'
    ]

    for pattern in m0_patterns:
        if re.search(pattern, text.lower()):
            if not re.search(r'(?:confirmed|definite|evident).*?metastasis', text.lower()):
                return {'classification': 'M0'}

    # M1a patterns: pleural/pericardial metastasis or contralateral lung nodules
    m1a_patterns = [
        r'(?:malignant|metastatic).*?(?:pleural|pericardial).*?(?:effusion|nodules?)',
        r'pleural.*?(?:metastases|metastasis|carcinomatosis)',
        r'pericardial.*?(?:metastases|metastasis|carcinomatosis)',
        r'separate.*?tumor.*?nodule.*?contralateral.*?lobe',
        r'malignant.*?(?:pleural|pericardial).*?effusion',
        r'(?:separate|additional).*?(?:nodule|lesion).*?different.*?lobe'
    ]

    for pattern in m1a_patterns:
        if re.search(pattern, text.lower()):
            return {'classification': 'M1a'}

    # M1b patterns: single extrathoracic metastasis
    m1b_patterns = [
        r'single.*?(?:distant|remote|extrathoracic).*?metastasis',
        r'solitary.*?metastasis',
        r'isolated.*?metastasis',
        r'oligometastasis.*?single.*?site',
        r'one.*?metastatic.*?lesion',
        r'single.*?(?:brain|liver|bone|adrenal).*?metastasis',
        r'solitary.*?(?:brain|liver|bone|adrenal).*?lesion.*?metastatic'
    ]

    for pattern in m1b_patterns:
        if re.search(pattern, text.lower()):
            return {'classification': 'M1b'}

    # M1c patterns: multiple extrathoracic metastases
    m1c_patterns = [
        r'multiple.*?(?:distant|organ).*?metastases',
        r'metastases.*?(?:multiple|several|many).*?(?:organs?|sites)',
        r'widespread.*?metastases',
        r'diffuse.*?metastases',
        r'metastases.*?(?:brain|liver|bone|adrenal).*?and.*?(?:brain|liver|bone|adrenal)',
        r'multiple.*?(?:brain|liver|bone|adrenal).*?lesions?.*?metastatic',
        r'(?:brain|liver|bone|adrenal).*?and.*?(?:brain|liver|bone|adrenal).*?metastases'
    ]

    for pattern in m1c_patterns:
        if re.search(pattern, text.lower()):
            return {'classification': 'M1c'}

    # Organ-specific patterns
    organ_specific_patterns = {
        'brain': [
            r'brain.*?(?:metastasis|metastases)',
            r'intracranial.*?(?:metastasis|metastases)',
            r'cerebral.*?(?:metastasis|metastases)'
        ],
        'bone': [
            r'bone.*?(?:metastasis|metastases)',
            r'skeletal.*?(?:metastasis|metastases)',
            r'osseous.*?(?:metastasis|metastases)'
        ],
        'liver': [
            r'liver.*?(?:metastasis|metastases)',
            r'hepatic.*?(?:metastasis|metastases)'
        ],
        'adrenal': [
            r'adrenal.*?(?:metastasis|metastases)',
            r'suprarenal.*?(?:metastasis|metastases)'
        ]
    }

    # Check for metastatic sites
    metastatic_sites = []
    for organ, patterns in organ_specific_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                metastatic_sites.append(organ)
                break

    # Classify based on number of metastatic sites
    if len(metastatic_sites) > 1:
        return {'classification': 'M1c'}
    elif len(metastatic_sites) == 1:
        return {'classification': 'M1b'}

    # Default: no clear evidence of distant metastasis
    return {'classification': 'M0'}

