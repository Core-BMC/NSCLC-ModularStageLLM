#!/usr/bin/env python3
"""Run TNM Classification Workflow from command line.

Usage:
    python run_workflow.py --i input/simple_case.xlsx --o output/simple_case_output
    python run_workflow.py -i input/difficult_case.csv -o output/difficult_case_output
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import Config
from src.tnm_workflow import TNMClassificationWorkflow
from src.utils.data_utils import process_excel_data
from src.utils.logging_utils import setup_logging
from src.utils.metrics_utils import calculate_metrics


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run TNM Classification Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_workflow.py --i input/simple_case.xlsx --o output/simple_case_output
  python run_workflow.py -i input/difficult_case.csv -o output/difficult_case_output
  python run_workflow.py --i input/sample_cases.csv --o output/results --config config/tnm_config.yaml
        """
    )
    
    parser.add_argument(
        '-i', '--i',
        dest='input_file',
        required=True,
        help='Input file path (CSV or Excel file)'
    )
    
    parser.add_argument(
        '-o', '--o',
        dest='output_prefix',
        required=True,
        help='Output file prefix (without extension). '
             'Output files will be: <prefix>.csv and <prefix>.json'
    )
    
    parser.add_argument(
        '--config',
        dest='config_path',
        default='config/tnm_config.yaml',
        help='Path to configuration YAML file (default: config/tnm_config.yaml)'
    )
    
    parser.add_argument(
        '--log',
        dest='log_file',
        default=None,
        help='Path to log file (default: <output_prefix>.log)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_output_paths(config: Config, output_prefix: str, log_file: str = None):
    """Setup output file paths in config.
    
    Args:
        config: Configuration object
        output_prefix: Output file prefix
        log_file: Optional log file path
    """
    # Ensure output directory exists
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output files
    csv_file = f"{output_prefix}.csv"
    json_file = f"{output_prefix}.json"
    
    config.set_input_file(config.get('input_file'))  # Keep original input_file
    config.config['output_file'] = csv_file
    config.config['json_output_file'] = json_file
    
    # Set log file
    if log_file:
        config.config['log_file'] = log_file
    else:
        config.config['log_file'] = f"{output_prefix}.log"
    
    logging.info(f"Output CSV: {csv_file}")
    logging.info(f"Output JSON: {json_file}")
    logging.info(f"Log file: {config.config['log_file']}")


def main():
    """Main entry point for workflow execution."""
    args = parse_arguments()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        # Load configuration
        config = Config(config_path=args.config_path)
        
        # Override input file in config
        config.set_input_file(args.input_file)
        
        # Setup logging
        logger = setup_logging(config)
        
        # Setup output paths
        setup_output_paths(config, args.output_prefix, args.log_file)
        
        # Set log level if verbose
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        logger.info("=" * 70)
        logger.info("TNM Classification Workflow")
        logger.info("=" * 70)
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output prefix: {args.output_prefix}")
        logger.info(f"Config file: {args.config_path}")
        logger.info(f"AJCC Edition: {config.get('ajcc_edition')} (AJCC {config.get('ajcc_version')} edition)")
        logger.info("=" * 70)
        
        # Initialize workflow
        workflow = TNMClassificationWorkflow(config)
        
        # Process input data
        logger.info("Processing input data")
        processed_data = process_excel_data(config)
        logger.info(f"Processed {len(processed_data)} cases from input file")
        
        if not processed_data:
            logger.warning("No cases found in input file")
            return
        
        # Process each case
        success_count = 0
        incomplete_count = 0
        error_count = 0
        
        try:
            for case in processed_data:
                case_number = case['case_number']
                logger.info(f"\n{'=' * 70}")
                logger.info(f"Processing case {case_number} ({case_number}/{len(processed_data)})")
                logger.info(f"{'=' * 70}")
                
                initial_state = {
                    "input": case['input'],
                    "true_tnm": case['true_tnm'],
                    "iteration_count": 0
                }
                
                final_state = workflow.run(initial_state)
                
                if final_state is None:
                    logger.warning(f"Failed to process case {case_number}")
                    error_count += 1
                elif final_state.get("next") is None:
                    logger.info(f"Case {case_number} processing completed successfully")
                    success_count += 1
                else:
                    logger.warning(
                        f"Case {case_number} processing ended at unexpected state: "
                        f"{final_state.get('next')}"
                    )
                    incomplete_count += 1
            
            # Finalize CSV file
            logger.info("\nFinalizing CSV file")
            workflow.finalize_csv()
            logger.info("All cases processed and final CSV file created")
            
            # Calculate and log metrics
            accuracy_metrics, confusion_metrics, detailed_results = calculate_metrics(workflow)
            
            if accuracy_metrics:
                logger.info("\n" + "=" * 70)
                logger.info("Classification Results Summary")
                logger.info("=" * 70)
                
                # Overall accuracy summary
                logger.info("\nOverall Performance:")
                for category in ['T', 'N', 'M', 'Stage']:
                    accuracy = accuracy_metrics.get(category, {})
                    if accuracy:
                        logger.info(
                            f"{category:<6} Accuracy: {accuracy.get('accuracy', 0):.2f}% "
                            f"({accuracy.get('correct', 0)}/{accuracy.get('total', 0)})"
                        )
                
                # Detailed analysis
                logger.info("\nDetailed Analysis by Category:")
                for category in ['T', 'N', 'M', 'Stage']:
                    accuracy = accuracy_metrics.get(category, {})
                    confusion = confusion_metrics.get(category, {})
                    
                    if accuracy and confusion:
                        logger.info(f"\n{category} Classification:")
                        logger.info(f"  Accuracy     : {accuracy.get('accuracy', 0):.2f}%")
                        logger.info(f"  Total Cases  : {accuracy.get('total', 0)}")
                        logger.info(f"  Correct      : {accuracy.get('correct', 0)}")
                        logger.info(f"  Over-staging : {confusion.get('over_staging', 0)} cases")
                        logger.info(f"  Under-staging: {confusion.get('under_staging', 0)} cases")
            else:
                logger.warning("No accuracy metrics available (true TNM values may be missing)")
            
            # Case-by-case details
            if detailed_results:
                logger.info("\n" + "=" * 70)
                logger.info("Case-by-Case Analysis")
                logger.info("=" * 70)
                for case in detailed_results:
                    logger.info(f"\nCase {case['case_id']} (Hospital #: {case['hospital_number']}):")
                    for category, comparison in case['comparisons'].items():
                        result = "✓" if comparison['correct'] else "✗"
                        logger.info(
                            f"  {category:<6}: {comparison['true']} -> "
                            f"{comparison['predicted']} {result}"
                        )
            
            # Processing summary
            logger.info("\n" + "=" * 70)
            logger.info("Processing Summary")
            logger.info("=" * 70)
            logger.info(f"Total Cases        : {len(processed_data)}")
            logger.info(f"Successfully Processed: {success_count}")
            logger.info(f"Incomplete Processing: {incomplete_count}")
            logger.info(f"Processing Errors  : {error_count}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.critical(f"Critical error in workflow execution: {str(e)}", exc_info=True)
            raise
        finally:
            if hasattr(workflow, 'temp_file') and workflow.temp_file:
                try:
                    workflow.temp_file.close()
                except Exception:
                    pass
        
        logger.info("\nTNM Classification Workflow completed successfully")
        logger.info(f"Results saved to: {args.output_prefix}.csv and {args.output_prefix}.json")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

