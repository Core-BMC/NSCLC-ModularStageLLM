"""Metrics calculation utilities for TNM staging workflow."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def analyze_staging_errors(predictions: List[Dict[str, str]]) -> Dict[str, int]:
    """Analyze staging errors to determine over-staging vs under-staging.

    Args:
        predictions: List of dictionaries containing 'true' and 'predicted' keys

    Returns:
        Dictionary with error analysis metrics
    """
    def get_stage_value(stage_str: str) -> Optional[float]:
        """Convert stage string to numeric value for comparison."""
        stage_values = {
            'T': {
                'T0': 0, 'Tis': 0.5, 'T1mi': 1, 'T1a': 1.1, 'T1b': 1.2,
                'T1c': 1.3, 'T2a': 2.1, 'T2b': 2.2, 'T3': 3, 'T4': 4
            },
            'N': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
            'M': {'M0': 0, 'M1a': 1.1, 'M1b': 1.2, 'M1c': 1.3},
            'Stage': {
                '0': 0, 'IA1': 1.1, 'IA2': 1.2, 'IA3': 1.3, 'IB': 1.4,
                'IIA': 2.1, 'IIB': 2.2, 'IIIA': 3.1, 'IIIB': 3.2,
                'IIIC': 3.3, 'IVA': 4.1, 'IVB': 4.2
            }
        }
        
        for category, values in stage_values.items():
            if stage_str in values:
                return values[stage_str]
        return None

    error_metrics = {
        'over_staging': 0,
        'under_staging': 0,
        'correct': 0,
        'total': len(predictions)
    }
    
    for pred in predictions:
        true_val = get_stage_value(pred['true'])
        pred_val = get_stage_value(pred['predicted'])
        
        if true_val is None or pred_val is None:
            continue
            
        if true_val == pred_val:
            error_metrics['correct'] += 1
        elif pred_val > true_val:
            error_metrics['over_staging'] += 1
        else:
            error_metrics['under_staging'] += 1
            
    return error_metrics


def calculate_metrics(
    workflow: Any
) -> Tuple[
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, int]]],
    Optional[List[Dict[str, Any]]]
]:
    """Calculate classification accuracy metrics.

    Args:
        workflow: Workflow instance with config attribute

    Returns:
        Tuple of (accuracy_metrics, confusion_metrics, detailed_results)
    """
    try:
        json_file_path = workflow.config.get(
            'json_output_file',
            'patient_data.json'
        )
        if not os.path.exists(json_file_path):
            logger.warning(f"JSON file not found: {json_file_path}")
            return None, None, None
            
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        if not results or not isinstance(results, list):
            logger.warning("No valid results found in JSON file")
            return None, None, None

        # Initialize metrics
        metrics = {
            'T': {'correct': 0, 'total': 0, 'predictions': []},
            'N': {'correct': 0, 'total': 0, 'predictions': []},
            'M': {'correct': 0, 'total': 0, 'predictions': []},
            'Stage': {'correct': 0, 'total': 0, 'predictions': []}
        }
        
        # Detailed results for analysis
        detailed_results = []
        
        # Calculate metrics for each case
        for case in results:
            true_tnm = case.get('tnm', {})
            ai_tnm = case.get('aiTnm', {})
            
            case_results = {
                'case_id': case.get('id'),
                'hospital_number': case.get('hospitalNumber'),
                'histology': case.get('histology', {}),
                'comparisons': {}
            }
            
            for category in ['T', 'N', 'M', 'Stage']:
                true_value = true_tnm.get(category)
                ai_value = ai_tnm.get(category)
                
                if true_value and ai_value:
                    metrics[category]['total'] += 1
                    metrics[category]['predictions'].append({
                        'true': true_value,
                        'predicted': ai_value
                    })
                    
                    is_correct = true_value == ai_value
                    if is_correct:
                        metrics[category]['correct'] += 1
                        
                    case_results['comparisons'][category] = {
                        'true': true_value,
                        'predicted': ai_value,
                        'correct': is_correct
                    }
            
            if any(case_results['comparisons']):
                detailed_results.append(case_results)
        
        # Calculate accuracy metrics
        accuracy_metrics = {}
        confusion_metrics = {}
        
        for category in metrics:
            if metrics[category]['total'] > 0:
                accuracy = (
                    metrics[category]['correct'] /
                    metrics[category]['total']
                ) * 100
                accuracy_metrics[category] = {
                    'accuracy': accuracy,
                    'correct': metrics[category]['correct'],
                    'total': metrics[category]['total']
                }
                
                predictions = metrics[category]['predictions']
                confusion_metrics[category] = analyze_staging_errors(
                    predictions
                )
        
        logger.info("Metrics calculation completed successfully")
        return accuracy_metrics, confusion_metrics, detailed_results
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return None, None, None

