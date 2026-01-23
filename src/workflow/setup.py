"""Workflow setup functions for TNM classification."""

import logging
from typing import Any, Dict

from langchain_core.tools import StructuredTool
from langgraph.graph import END

from src.parsers import (
    TNM_M_Parser,
    TNM_N_Parser,
    TNM_T_Parser,
    TNMOutputParser,
)

logger = logging.getLogger(__name__)


def setup_prompts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup classification prompts with proper formatting.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with prompt keys and consensus flags:
        - t_classifier_prompt, n_classifier_prompt, m_classifier_prompt
        - t_use_consensus, n_use_consensus, m_use_consensus
        - tnm_classifier_prompt, tnm_use_consensus

    Raises:
        ValueError: If prompts are missing
    """
    try:
        # Determine AJCC edition from config
        ajcc_edition = config.get('ajcc_edition', 'ajcc8th')
        ajcc_version = config.get('ajcc_version', '8th')
        
        # Load prompts based on AJCC edition
        # New structure: ajcc8th_prompts / ajcc9th_prompts
        # Fallback to old structure: prompts (for backward compatibility)
        if ajcc_edition == 'ajcc9th':
            prompts_key = 'ajcc9th_prompts'
        else:
            prompts_key = 'ajcc8th_prompts'
        
        prompts = config.get(prompts_key, {})
        
        # Ensure prompts is a dictionary, not a Config object
        if not isinstance(prompts, dict):
            prompts = {}
        
        # Fallback to old structure if new structure not found
        if not prompts:
            prompts = config.get('prompts', {})
            # Ensure prompts is a dictionary
            if not isinstance(prompts, dict):
                prompts = {}
            if not prompts:
                logger.warning(
                    f"{prompts_key} not found, using legacy 'prompts' structure. "
                    f"Consider updating config to use {prompts_key}."
                )
        
        tnm_base = bool(config.get('tnm_base', False))
        # Check global use_consensus setting (takes priority)
        global_use_consensus = config.get('use_consensus', None)

        if tnm_base:
            tnm_classifier_prompt = prompts.get('tnm_classifier_base', '')
            if not tnm_classifier_prompt:
                available_keys = list(prompts.keys())
                raise ValueError(
                    f"Missing tnm_classifier_base prompt in config file. "
                    f"Available keys: {available_keys}"
                )

            logger.info(
                f"Loading base TNM prompt (AJCC {ajcc_version} edition):"
            )
            prompts_source = prompts_key if config.get(prompts_key) else 'prompts (legacy)'
            logger.info(f"Using prompts from: {prompts_source}")
            logger.info(f"Detected AJCC edition: {ajcc_edition} (from config.get('ajcc_edition'))")
            if global_use_consensus is not None:
                logger.info(
                    "Global consensus setting is ignored for tnm_base workflow"
                )
            logger.debug(
                f"TNM Base Prompt length: {len(tnm_classifier_prompt)} chars"
            )

            return {
                'tnm_classifier_prompt': tnm_classifier_prompt,
                'tnm_use_consensus': False,
                't_classifier_prompt': prompts.get('t_classifier', ''),
                'n_classifier_prompt': prompts.get('n_classifier', ''),
                'm_classifier_prompt': prompts.get('m_classifier', ''),
                't_use_consensus': False,
                'n_use_consensus': False,
                'm_use_consensus': False,
            }
        
        # Determine which prompts to use and consensus settings
        # Priority: 1) global use_consensus setting, 2) _classifier (with consensus) > _classifier_base (no consensus)
        t_classifier_prompt = prompts.get('t_classifier', '')
        t_classifier_base = prompts.get('t_classifier_base', '')
        
        if global_use_consensus is not None:
            # Use global setting
            t_use_consensus = global_use_consensus
            # Prefer _classifier if available, otherwise use _classifier_base
            t_final_prompt = t_classifier_prompt if t_classifier_prompt else t_classifier_base
        else:
            # Fallback to prompt-based detection
            t_use_consensus = bool(t_classifier_prompt)
            t_final_prompt = t_classifier_prompt if t_use_consensus else t_classifier_base

        n_classifier_prompt = prompts.get('n_classifier', '')
        n_classifier_base = prompts.get('n_classifier_base', '')
        
        if global_use_consensus is not None:
            n_use_consensus = global_use_consensus
            n_final_prompt = n_classifier_prompt if n_classifier_prompt else n_classifier_base
        else:
            n_use_consensus = bool(n_classifier_prompt)
            n_final_prompt = n_classifier_prompt if n_use_consensus else n_classifier_base

        m_classifier_prompt = prompts.get('m_classifier', '')
        m_classifier_base = prompts.get('m_classifier_base', '')
        
        if global_use_consensus is not None:
            m_use_consensus = global_use_consensus
            m_final_prompt = m_classifier_prompt if m_classifier_prompt else m_classifier_base
        else:
            m_use_consensus = bool(m_classifier_prompt)
            m_final_prompt = m_classifier_prompt if m_use_consensus else m_classifier_base

        # Validate prompts exist
        if not all([t_final_prompt, n_final_prompt, m_final_prompt]):
            available_keys = list(prompts.keys())
            raise ValueError(
                f"Missing prompts in config file. "
                f"Required: t_classifier/t_classifier_base, "
                f"n_classifier/n_classifier_base, "
                f"m_classifier/m_classifier_base. "
                f"Available keys: {available_keys}"
            )

        # Log the prompts being loaded
        logger.info(f"Loading classification prompts (AJCC {ajcc_version} edition):")
        # Check if prompts_key exists in config (use get() to avoid 'in' operator on Config object)
        prompts_source = prompts_key if config.get(prompts_key) else 'prompts (legacy)'
        logger.info(f"Using prompts from: {prompts_source}")
        logger.info(f"Detected AJCC edition: {ajcc_edition} (from config.get('ajcc_edition'))")
        
        # Verify which prompts were actually loaded
        n_has_n2a_n2b = 'N2a' in n_final_prompt or 'N2b' in n_final_prompt
        m_has_m1c1_m1c2 = 'M1c1' in m_final_prompt or 'M1c2' in m_final_prompt
        logger.info(f"Prompt verification - N prompt has N2a/N2b: {n_has_n2a_n2b}, M prompt has M1c1/M1c2: {m_has_m1c1_m1c2}")
        
        if global_use_consensus is not None:
            logger.info(f"Global consensus setting: {global_use_consensus}")
        logger.info(
            f"T Classifier: {'CONSENSUS MODE' if t_use_consensus else 'SINGLE RESPONSE MODE'}"
        )
        logger.info(
            f"N Classifier: {'CONSENSUS MODE' if n_use_consensus else 'SINGLE RESPONSE MODE'}"
        )
        logger.info(
            f"M Classifier: {'CONSENSUS MODE' if m_use_consensus else 'SINGLE RESPONSE MODE'}"
        )
        logger.debug(f"T Classifier Prompt length: {len(t_final_prompt)} chars")
        logger.debug(f"N Classifier Prompt length: {len(n_final_prompt)} chars")
        logger.debug(f"M Classifier Prompt length: {len(m_final_prompt)} chars")

        return {
            't_classifier_prompt': t_final_prompt,
            'n_classifier_prompt': n_final_prompt,
            'm_classifier_prompt': m_final_prompt,
            't_use_consensus': t_use_consensus,
            'n_use_consensus': n_use_consensus,
            'm_use_consensus': m_use_consensus,
            'tnm_classifier_prompt': '',
            'tnm_use_consensus': False,
        }

    except Exception as e:
        logger.error(f"Error setting up prompts: {e}")
        raise


def setup_agents() -> Dict[str, Any]:
    """Setup classification agents with proper prompts.

    Returns:
        Dictionary with tools and parsers (tools, t_parser, n_parser, m_parser)

    Raises:
        Exception: If agent setup fails
    """
    try:
        def classify_tnm(
            t_stage: str,
            n_stage: str,
            m_stage: str,
            reasoning: str
        ) -> dict:
            return {
                "t_stage": t_stage,
                "n_stage": n_stage,
                "m_stage": m_stage,
                "reasoning": reasoning
            }

        tnm_tool = StructuredTool.from_function(
            func=classify_tnm,
            name="classify_tnm",
            description=(
                "Classify the TNM stage of lung cancer based on "
                "given information"
            )
        )

        tools = [tnm_tool]

        # Setup parsers
        t_parser = TNM_T_Parser()
        n_parser = TNM_N_Parser()
        m_parser = TNM_M_Parser()
        tnm_parser = TNMOutputParser()

        return {
            'tools': tools,
            'tnm_tool': tnm_tool,
            't_parser': t_parser,
            'n_parser': n_parser,
            'm_parser': m_parser,
            'tnm_parser': tnm_parser
        }

    except Exception as e:
        logger.error(f"Error setting up agents: {e}")
        raise


def setup_graph(
    workflow,
    node_functions: Dict[str, Any],
    tnm_base: bool = False
):
    """Setup workflow graph with nodes and edges.

    Args:
        workflow: StateGraph instance
        node_functions: Dictionary with node function references:
            - histology_classifier_node
            - tnm_classifier_node
            - t_classifier_node
            - n_classifier_node
            - m_classifier_node
            - stage_classifier_node
            - final_save_node

    Raises:
        Exception: If graph setup fails
    """
    try:
        # Add nodes
        workflow.add_node(
            "histology_classifier",
            node_functions['histology_classifier_node']
        )
        workflow.add_node(
            "stage_classifier",
            node_functions['stage_classifier_node']
        )
        workflow.add_node("final_save", node_functions['final_save_node'])

        if tnm_base:
            workflow.add_node(
                "tnm_classifier",
                node_functions['tnm_classifier_node']
            )
            workflow.add_edge("histology_classifier", "tnm_classifier")
            workflow.add_edge("tnm_classifier", "stage_classifier")
        else:
            workflow.add_node(
                "t_classifier",
                node_functions['t_classifier_node']
            )
            workflow.add_node(
                "n_classifier",
                node_functions['n_classifier_node']
            )
            workflow.add_node(
                "m_classifier",
                node_functions['m_classifier_node']
            )
            workflow.add_edge("histology_classifier", "t_classifier")
            workflow.add_edge("t_classifier", "n_classifier")
            workflow.add_edge("n_classifier", "m_classifier")
            workflow.add_edge("m_classifier", "stage_classifier")

        workflow.add_edge("stage_classifier", "final_save")
        workflow.add_edge("final_save", END)

        # Set entry point to histology_classifier
        workflow.set_entry_point("histology_classifier")

    except Exception as e:
        logger.error(f"Failed to setup workflow graph: {e}")
        raise
