"""Error handling utilities for TNM classification workflow."""

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from src.workflow import AgentState

logger = logging.getLogger(__name__)


def create_error_state(
    state: AgentState,
    node_name: str,
    error_message: str,
    next_node: Optional[str]
) -> AgentState:
    """Create error state for workflow.

    Args:
        state: Current workflow state
        node_name: Name of the node where error occurred
        error_message: Error message
        next_node: Next node to transition to (None for END)

    Returns:
        Updated state with error information
    """
    logger.error(
        f"Error in {node_name} for case "
        f"{state['input'].get('case_number')}: {error_message}"
    )
    state['input']['error_count'] = state['input'].get('error_count', 0) + 1

    # Preserve True TNM data
    true_tnm = state.get('true_tnm', {})

    if state['input']['error_count'] > 2:
        logger.critical(
            f"Max error count reached for case "
            f"{state['input'].get('case_number')}. Terminating process."
        )
        return {
            "messages": state.get("messages", []) + [
                HumanMessage(
                    content="Max error count reached. Terminating process.",
                    name=node_name
                )
            ],
            "intermediate_steps": state.get('intermediate_steps', []),
            "input": state['input'],
            "true_tnm": true_tnm,
            "next": END
        }

    logger.warning(
        f"Attempting to continue to {next_node} after error in {node_name}"
    )
    return {
        "messages": state.get("messages", []) + [
            HumanMessage(
                content=f"Error occurred: {error_message}",
                name=node_name
            )
        ],
        "intermediate_steps": state.get('intermediate_steps', []),
        "input": state['input'],
        "true_tnm": true_tnm,
        "next": next_node
    }


def retry_node(
    node_func,
    state: AgentState,
    max_retries: int = 2
) -> AgentState:
    """Retry a node function with error handling.

    Args:
        node_func: Node function to retry
        state: Current workflow state
        max_retries: Maximum number of retries

    Returns:
        Updated state after retry attempts
    """
    for attempt in range(max_retries + 1):
        try:
            return node_func(state)
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1} failed. Retrying..."
                )
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed. "
                    f"Recording error and moving to next node."
                )
                return record_error(state, str(e), node_func.__name__)


def record_error(
    state: AgentState,
    error_message: str,
    node_name: str
) -> AgentState:
    """Record error in state.

    Args:
        state: Current workflow state
        error_message: Error message
        node_name: Name of the node where error occurred

    Returns:
        Updated state with error information
    """
    error_output = {
        "classification": "Error"
    }
    state[f'{node_name}_classification'] = error_output
    return state


def extract_json_from_result(
    result: Any,
    parser: Any
) -> Dict[str, Any]:
    """Parse result using output parser.

    Args:
        result: Result to parse
        parser: Parser instance

    Returns:
        Parsed dictionary or empty dict on error
    """
    try:
        if isinstance(result, dict) and 'output' in result:
            return parser.parse(result['output'])
        return parser.parse(str(result))
    except Exception as e:
        logger.error(f"Error parsing result with output parser: {e}")
        return {}


def extract_json_using_regex(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text using regex pattern.

    Args:
        text: Text to extract JSON from

    Returns:
        Extracted JSON object or None
    """
    json_pattern = (
        r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
    )
    match = re.search(json_pattern, text)
    if match:
        try:
            json_obj = json.loads(match.group())
            logger.debug(
                f"Successfully extracted JSON using regex: "
                f"{json.dumps(json_obj, indent=2)}"
            )
            return json_obj
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing error after regex extraction: {str(e)}"
            )
    return None

