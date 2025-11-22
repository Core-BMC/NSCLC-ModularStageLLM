"""Stage determination utilities for TNM classification."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def determine_stage_from_tnm(
    t_class: str,
    n_class: str,
    m_class: str,
    stage_rules: Dict[str, Any]
) -> str:
    """Determine stage based on TNM values according to rules from config.

    Args:
        t_class: T classification (e.g., 'T1a', 'T2b', 'T3', 'T4')
        n_class: N classification (e.g., 'N0', 'N1', 'N2', 'N3', 'N2a', 'N2b')
        m_class: M classification (e.g., 'M0', 'M1a', 'M1b', 'M1c')
        stage_rules: Stage determination rules from config

    Returns:
        Stage classification string (e.g., 'IA1', 'IIIA', 'IVB')
    """
    try:
        if not stage_rules:
            logger.warning(
                "No stage_rules found in config, using fallback rules"
            )
            return determine_stage_fallback(t_class, n_class, m_class)

        # Check M1 category (Stage IV) - highest priority
        if m_class.startswith('M1'):
            m_categories = stage_rules.get('m_categories', {})

            # Handle M1c subcategories (M1c1, M1c2 for 9th edition)
            if m_class.startswith('M1c'):
                # Check for specific M1c subcategories first
                if m_class in m_categories:
                    return m_categories[m_class]
                # Fallback to generic M1c
                return m_categories.get('M1c', 'IVB')

            # Handle M1a and M1b
            return m_categories.get(m_class, 'IVA')

        # Check N3
        if n_class == 'N3':
            n3_rules = stage_rules.get('n3_rules', {})
            if t_class in ['T3', 'T4']:
                return n3_rules.get('T3_T4', 'IIIC')
            return n3_rules.get('default', 'IIIB')

        # Check N2 (including N2a, N2b for 9th edition)
        if n_class.startswith('N2'):
            n2_rules = stage_rules.get('n2_rules', {})

            # Handle N2a/N2b subcategories (9th edition)
            if n_class in ['N2a', 'N2b']:
                n2_subcategory_rules = n2_rules.get(n_class, {})
                if isinstance(n2_subcategory_rules, dict):
                    # Check T-specific rules for N2a/N2b
                    if t_class in n2_subcategory_rules:
                        return n2_subcategory_rules[t_class]
                    return n2_subcategory_rules.get('default', 'IIIA')

            # Handle generic N2 (8th edition or fallback)
            if t_class in ['T3', 'T4']:
                return n2_rules.get('T3_T4', 'IIIB')

            # Check T1_T2 rules
            t1_t2_rules = n2_rules.get('T1_T2', {})
            if isinstance(t1_t2_rules, dict):
                return t1_t2_rules.get(t_class, 'IIIA')

            return 'IIIA'

        # Check N1
        if n_class == 'N1':
            n1_rules = stage_rules.get('n1_rules', {})

            if t_class == 'T3':
                return n1_rules.get('T3', 'IIIA')
            elif t_class == 'T4':
                return n1_rules.get('T4', 'IIIA')

            # Check T1_T2 rules
            t1_t2_rules = n1_rules.get('T1_T2', {})
            if isinstance(t1_t2_rules, dict):
                return t1_t2_rules.get(t_class, 'IIB')

            return 'IIB'

        # Check N0
        if n_class == 'N0':
            n0_rules = stage_rules.get('n0_rules', {})
            return n0_rules.get(t_class, 'Unknown')

        return 'Unknown'

    except Exception as e:
        logger.error(f"Error in stage determination: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 'Unknown'


def determine_stage_fallback(
    t_class: str,
    n_class: str,
    m_class: str
) -> str:
    """Fallback stage determination when config rules are not available.

    This function provides default AJCC 8th edition rules as a fallback.

    Args:
        t_class: T classification
        n_class: N classification
        m_class: M classification

    Returns:
        Stage classification string
    """
    # Default AJCC 8th edition rules
    if m_class.startswith('M1'):
        if m_class.startswith('M1c'):
            return 'IVB'
        return 'IVA'

    if n_class == 'N3':
        return 'IIIC' if t_class in ['T3', 'T4'] else 'IIIB'

    if n_class == 'N2':
        if t_class in ['T3', 'T4']:
            return 'IIIB'
        return 'IIIA'

    if n_class == 'N1':
        if t_class in ['T3', 'T4']:
            return 'IIIA'
        return 'IIB'

    if n_class == 'N0':
        n0_map = {
            'Tis': '0',
            'T1mi': 'IA1',
            'T1a': 'IA1',
            'T1b': 'IA2',
            'T1c': 'IA3',
            'T2a': 'IB',
            'T2b': 'IIA',
            'T3': 'IIB',
            'T4': 'IIIA'
        }
        return n0_map.get(t_class, 'Unknown')

    return 'Unknown'


def validate_stage_result(stage: str, t: str, n: str, m: str) -> bool:
    """Validate stage determination.

    Args:
        stage: Determined stage
        t: T classification
        n: N classification
        m: M classification

    Returns:
        True if stage is valid, False otherwise
    """
    valid_stages = {
        '0', 'IA1', 'IA2', 'IA3', 'IB', 'IIA', 'IIB',
        'IIIA', 'IIIB', 'IIIC', 'IVA', 'IVB'
    }

    if stage not in valid_stages:
        logger.error(f"Invalid stage determined: {stage}")
        return False

    # M1 cases should always be stage IV
    if m.startswith('M1') and not stage.startswith('IV'):
        logger.error(f"Inconsistent stage {stage} for M1 disease")
        return False

    return True


def get_stage_explanation(stage: str, t: str, n: str, m: str) -> str:
    """Generate detailed explanation for stage determination.

    Args:
        stage: Stage classification
        t: T classification
        n: N classification
        m: M classification

    Returns:
        Detailed explanation string
    """
    explanations = {
        '0': (
            "Carcinoma in situ (Tis) with no spread to lymph nodes "
            "or distant sites"
        ),
        'IA1': (
            "Early-stage disease with T1mi (minimally invasive) or T1a "
            "tumor (≤1 cm) and no lymph node involvement"
        ),
        'IA2': (
            "Early-stage disease with T1b tumor (>1 cm but ≤2 cm) "
            "and no lymph node involvement"
        ),
        'IA3': (
            "Early-stage disease with T1c tumor (>2 cm but ≤3 cm) "
            "and no lymph node involvement"
        ),
        'IB': (
            "Relatively early-stage disease with T2a tumor "
            "(>3 cm but ≤4 cm) and no lymph node involvement"
        ),
        'IIA': (
            "Local disease with T2b tumor (>4 cm but ≤5 cm) "
            "and no lymph node involvement"
        ),
        'IIB': (
            "Locally advanced disease with either T3 tumor or T1-T2 "
            "tumor with N1 lymph node involvement"
        ),
        'IIIA': (
            "Advanced local disease with T1-T2/N2, T3/N1, "
            "or T4/N0-N1 involvement"
        ),
        'IIIB': (
            "Advanced disease with T1-T2/N3, T3-T4/N2, "
            "or T3-T4/N3 involvement"
        ),
        'IIIC': (
            "Very advanced local disease with T3-T4 tumor "
            "and N3 lymph node involvement"
        ),
        'IVA': (
            "Metastatic disease with M1a (contralateral lung nodules "
            "or pleural/pericardial dissemination) or M1b "
            "(single extrathoracic metastasis)"
        ),
        'IVB': (
            "Extensively metastatic disease (M1c) with multiple "
            "extrathoracic metastases"
        )
    }

    base_explanation = explanations.get(
        stage, "Stage based on TNM combination"
    )
    specific_factors = []

    if m.startswith('M1'):
        specific_factors.append(f"presence of {m} metastatic disease")
    elif t.startswith('T4'):
        specific_factors.append("extensive local tumor involvement")

    if specific_factors:
        base_explanation += (
            f" specifically due to {', '.join(specific_factors)}"
        )

    return base_explanation

