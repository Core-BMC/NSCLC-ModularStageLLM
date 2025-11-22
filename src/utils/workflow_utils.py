"""Workflow utility functions for TNM classification."""

from typing import Any, Dict


def prepare_node_specific_input(
    input_data: Dict[str, Any],
    node_type: str
) -> Dict[str, Any]:
    """Prepare input data with specific report ordering for each node type.

    Args:
        input_data: Original input data dictionary
        node_type: Type of node (t, n, or m)

    Returns:
        Ordered input data dictionary
    """
    # Define report order for each node type
    report_orders = {
        "t": [
            "chest_ct", "pathology", "pet", "brain_mr", "ebus",
            "neck_biopsy", "bone_scan", "abdomen_pelvis_ct", "adrenal_ct"
        ],
        "n": [
            "pathology", "ebus", "brain_mr", "chest_ct", "pet",
            "neck_biopsy", "bone_scan", "abdomen_pelvis_ct", "adrenal_ct"
        ],
        "m": [
            "pathology", "brain_mr", "bone_scan", "chest_ct",
            "abdomen_pelvis_ct", "adrenal_ct", "pet", "neck_biopsy", "ebus"
        ]
    }

    # Preserve basic information
    ordered_input = {
        "case_number": input_data["case_number"],
        "hospital_id": input_data["hospital_id"]
    }

    # Determine report order based on node type
    if node_type.lower() in report_orders:
        order = report_orders[node_type.lower()]

        # Add reports in specified order
        for field in order:
            if field in input_data and input_data[field] is not None:
                ordered_input[field] = input_data[field]
    else:
        # Unknown node type: preserve original order
        ordered_input.update({
            k: v for k, v in input_data.items()
            if k not in ["case_number", "hospital_id"]
        })

    return ordered_input

