"""TNM Staging Workflow for Non-Small Cell Lung Cancer.

This module provides a modular agent-based system for TNM staging
classification using Large Language Models (LLMs).
"""

import csv
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# Import models
from src.models import Config

# Import parsers
from src.parsers import (
    TNM_M_Parser,
    TNM_N_Parser,
    TNM_T_Parser,
    TNMOutputParser,
)

# Import agents
from src.agents import AgentConsensus

# Import workflow state
from src.workflow import AgentState

# Import utility functions
from src.utils.data_utils import format_input_data
from src.utils.error_utils import create_error_state, extract_json_using_regex
from src.utils.file_operations import (
    finalize_csv,
    save_to_csv_file,
    save_to_json_file,
)
from src.utils.llm_utils import get_llm_instance, setup_llm
from src.utils.logging_utils import setup_logging
from src.utils.metrics_utils import calculate_metrics
from src.utils.stage_utils import determine_stage_from_tnm
from src.utils.tnm_config_utils import filter_special_notes
from src.utils.workflow_utils import prepare_node_specific_input

# Import data processing utilities
from src.utils.data_utils import process_excel_data

# Import workflow setup functions
from src.workflow.setup import setup_agents, setup_graph, setup_prompts

# Import histology modules
from src.histology import HistologyClassificationWorkflow

logger = logging.getLogger(__name__)

# Models, Parsers, Agents, and State are now imported from separate modules
# MedicalReport, InputData, Config -> src/models/
# BaseTNMParser, TNM_T_Parser, TNM_N_Parser, TNM_M_Parser, TNMOutputParser -> src/parsers/
# AgentConsensus -> src/agents/
# AgentState -> src/workflow/state.py

# Removed duplicate class definitions - now imported from:
# TNMClassificationDict, BaseTNMParser -> src/parsers/base_parser.py
# TNM_T_Parser, TNM_N_Parser, TNM_M_Parser, TNMOutputParser -> src/parsers/tnm_parsers.py
# AgentConsensus -> src/agents/consensus.py
# AgentState -> src/workflow/state.py


class TNMClassificationWorkflow:
    """Main workflow for TNM classification."""

    def __init__(self, config: Config):
        """Initialize TNM classification workflow."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tnm_criteria = self.config.get('tnm_criteria', {})
        self.histology_workflow = HistologyClassificationWorkflow(config)
        self.tnm_base = bool(self.config.get('tnm_base', False))
        
        # Initialize LLM settings before setting up prompts
        self.llm_default = setup_llm(self.config)
        
        # Setup prompts using utility function
        prompts_dict = setup_prompts(self.config)
        self.t_classifier_prompt = prompts_dict['t_classifier_prompt']
        self.n_classifier_prompt = prompts_dict['n_classifier_prompt']
        self.m_classifier_prompt = prompts_dict['m_classifier_prompt']
        self.t_use_consensus = prompts_dict.get('t_use_consensus', True)
        self.n_use_consensus = prompts_dict.get('n_use_consensus', True)
        self.m_use_consensus = prompts_dict.get('m_use_consensus', True)
        self.tnm_classifier_prompt = prompts_dict.get('tnm_classifier_prompt', '')
        self.tnm_use_consensus = prompts_dict.get('tnm_use_consensus', False)
        
        # Setup agents using utility function
        agents_dict = setup_agents()
        self.tools = agents_dict['tools']
        self.tnm_tool = agents_dict['tnm_tool']
        self.t_parser = agents_dict['t_parser']
        self.n_parser = agents_dict['n_parser']
        self.m_parser = agents_dict['m_parser']
        self.tnm_parser = agents_dict['tnm_parser']
        
        # Setup workflow graph
        self.workflow = StateGraph(AgentState)
        setup_graph(
            self.workflow,
            {
                'histology_classifier_node': self.histology_classifier_node,
                'tnm_classifier_node': self.tnm_classifier_node,
                't_classifier_node': self.t_classifier_node,
                'n_classifier_node': self.n_classifier_node,
                'm_classifier_node': self.m_classifier_node,
                'stage_classifier_node': self.stage_classifier_node,
                'final_save_node': self.final_save_node
            },
            tnm_base=self.tnm_base
        )
        self.consensus_agent = AgentConsensus(max_retries=4)        
        
        self.fieldnames = [
            'pid', 'hospital_number', 
            'histology_category', 'histology_subcategory', 'histology_type', 'histology_confidence', 'histology_reason',
            'true_T', 'true_N', 'true_M', 'true_Stage',  
            'T_classification', 'N_classification', 'M_classification', 'Stage_classification',
            'T_reasoning', 'N_reasoning', 'M_reasoning', 'Stage_reasoning'
        ]
        
        self.temp_file_path = os.path.join(tempfile.gettempdir(), 'tnm_temp.csv')
        self.temp_file = open(self.temp_file_path, 'w+', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.temp_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()

    def _load_tnm_json(self) -> Dict[str, Any]:
        try:
            tnm_json_path = self.config.get('tnm_json_path', 'tnm_classification.json')
            with open(tnm_json_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Failed to load TNM JSON: {e}")
            raise

    def _get_llm_instance(self):
        """Get LLM instance with random temperature"""
        return get_llm_instance(self.config)


    def histology_classifier_node(self, state: AgentState) -> AgentState:
        """Histology classification node"""
        try:
            self.logger.debug(f"Histology classifier processing case {state['input'].get('case_number')}")
            
            # Get histology classification
            histology_result = self.histology_workflow.histology_classifier_node(state)
            
            # Add histology result to state
            state['input']['histology_classification'] = {
                'category': histology_result['histology_category'],
                'subcategory': histology_result.get('histology_subcategory', ''),
                'type': histology_result['histology_type'],
                'reasoning': histology_result['histology_reason']
            }
            
            next_node = "tnm_classifier" if self.tnm_base else "t_classifier"
            result_state = {
                **state,
                "next": next_node,
                "histology_category": histology_result.get("histology_category"),
                "histology_subcategory": histology_result.get("histology_subcategory", ''),
                "histology_type": histology_result.get("histology_type"),
                "histology_confidence": histology_result.get("histology_confidence"),
                "histology_reason": histology_result.get("histology_reason")
            }

            return result_state

        except Exception as e:
            self.logger.error(f"Error in Histology Classifier: {str(e)}", exc_info=True)
            next_node = "tnm_classifier" if self.tnm_base else "t_classifier"
            return create_error_state(state, "histology_classifier", str(e), next_node)

    def t_classifier_node(self, state: AgentState) -> AgentState:
        """T classification node passing workflow instance"""
        self.logger.debug(f"T classifier processing case {state['input'].get('case_number')}")
        try:
            # Reorder input data for T node
            ordered_input = prepare_node_specific_input(
                state['input'], 't'
            )

            # Use consensus agent with workflow instance (self)
            # Use consensus if t_classifier prompt is used, otherwise single response
            consensus_result = self.consensus_agent.collect_responses(
                self,  # Pass workflow instance instead of agent_executor
                {"input": ordered_input},
                self.t_parser,
                "T",
                use_consensus=self.t_use_consensus
            )

            if not consensus_result:
                raise ValueError("Failed to reach consensus on T classification")

            # Update state with consensus result including reasoning
            t_reasoning = consensus_result.get('reasoning', '')
            self.logger.info(f"T classification reasoning extracted (length: {len(t_reasoning)}): {t_reasoning[:100] if t_reasoning else 'EMPTY'}")
            state['input']['t_classification'] = {
                'classification': consensus_result['classification'],
                'raw_output': consensus_result.get('raw_output', ''),
                'reasoning': t_reasoning
            }
            
            result_state = {
                **state,
                "next": "n_classifier",
                "t_classification": consensus_result['classification']
            }

            return result_state

        except Exception as e:
            self.logger.error(f"Error in T Classifier: {str(e)}", exc_info=True)
            return create_error_state(state, "t_classifier", str(e), "n_classifier")

    def tnm_classifier_node(self, state: AgentState) -> AgentState:
        """TNM base classification node (single-pass T+N+M)."""
        self.logger.debug(f"TNM classifier processing case {state['input'].get('case_number')}")
        try:
            if not self.tnm_classifier_prompt:
                raise ValueError("TNM classifier base prompt not found")

            ordered_input = prepare_node_specific_input(
                state['input'], 'tnm'
            )

            input_text = format_input_data(ordered_input)
            system_message = SystemMessage(content=self.tnm_classifier_prompt)
            human_message = HumanMessage(
                content="Please provide the TNM classification:\n\n"
                f"{input_text}"
            )

            llm = self._get_llm_instance()
            response = llm.invoke([system_message, human_message])
            if not response:
                raise ValueError("Empty response from TNM classifier")

            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, str):
                response_content = response
            elif isinstance(response, dict):
                response_content = response.get('content', str(response))
            else:
                response_content = str(response)

            if not response_content:
                raise ValueError("Empty response content from TNM classifier")

            parsed_result = self.tnm_parser.parse(response_content)

            json_payload = extract_json_using_regex(response_content) or {}
            t_reasoning = (
                json_payload.get('t_reasoning', '')
                if isinstance(json_payload.get('t_reasoning', ''), str)
                else ''
            )
            n_reasoning = (
                json_payload.get('n_reasoning', '')
                if isinstance(json_payload.get('n_reasoning', ''), str)
                else ''
            )
            m_reasoning = (
                json_payload.get('m_reasoning', '')
                if isinstance(json_payload.get('m_reasoning', ''), str)
                else ''
            )

            state['input']['t_classification'] = {
                'classification': parsed_result['t_classification'],
                'raw_output': response_content,
                'reasoning': t_reasoning.strip()
            }
            state['input']['n_classification'] = {
                'classification': parsed_result['n_classification'],
                'raw_output': response_content,
                'reasoning': n_reasoning.strip()
            }
            state['input']['m_classification'] = {
                'classification': parsed_result['m_classification'],
                'raw_output': response_content,
                'reasoning': m_reasoning.strip()
            }

            result_state = {
                **state,
                "next": "stage_classifier",
                "t_classification": parsed_result['t_classification'],
                "n_classification": parsed_result['n_classification'],
                "m_classification": parsed_result['m_classification']
            }

            return result_state

        except Exception as e:
            self.logger.error(f"Error in TNM Classifier: {str(e)}", exc_info=True)
            return create_error_state(state, "tnm_classifier", str(e), "stage_classifier")

    def n_classifier_node(self, state: AgentState) -> AgentState:
        """N classification node passing workflow instance"""
        self.logger.debug(f"N classifier processing case {state['input'].get('case_number')}")
        try:
            ordered_input = prepare_node_specific_input(state['input'], 'n')
            
            # Use consensus agent with workflow instance
            # Use consensus if n_classifier prompt is used, otherwise single response
            consensus_result = self.consensus_agent.collect_responses(
                self,  # Pass workflow instance
                {"input": ordered_input},
                self.n_parser,
                "N",
                use_consensus=self.n_use_consensus
            )

            if not consensus_result:
                raise ValueError("Failed to reach consensus on N classification")

            n_reasoning = consensus_result.get('reasoning', '')
            self.logger.info(f"N classification reasoning extracted (length: {len(n_reasoning)}): {n_reasoning[:100] if n_reasoning else 'EMPTY'}")
            state['input']['n_classification'] = {
                'classification': consensus_result['classification'],
                'raw_output': consensus_result.get('raw_output', ''),
                'reasoning': n_reasoning
            }
            
            result_state = {
                **state,
                "next": "m_classifier",
                "n_classification": consensus_result['classification']
            }

            return result_state

        except Exception as e:
            self.logger.error(f"Error in N Classifier: {str(e)}", exc_info=True)
            return create_error_state(state, "n_classifier", str(e), "m_classifier")

    def m_classifier_node(self, state: AgentState) -> AgentState:
        """M classification node passing workflow instance"""
        self.logger.debug(f"M classifier processing case {state['input'].get('case_number')}")
        try:
            ordered_input = prepare_node_specific_input(state['input'], 'm')
            
            # Use consensus agent with workflow instance
            # Use consensus if m_classifier prompt is used, otherwise single response
            consensus_result = self.consensus_agent.collect_responses(
                self,  # Pass workflow instance
                {"input": ordered_input},
                self.m_parser,
                "M",
                use_consensus=self.m_use_consensus
            )

            if not consensus_result:
                raise ValueError("Failed to reach consensus on M classification")

            m_reasoning = consensus_result.get('reasoning', '')
            self.logger.info(f"M classification reasoning extracted (length: {len(m_reasoning)}): {m_reasoning[:100] if m_reasoning else 'EMPTY'}")
            state['input']['m_classification'] = {
                'classification': consensus_result['classification'],
                'raw_output': consensus_result.get('raw_output', ''),
                'reasoning': m_reasoning
            }
            
            result_state = {
                **state,
                "next": "stage_classifier",
                "m_classification": consensus_result['classification']
            }

            return result_state

        except Exception as e:
            self.logger.error(f"Error in M Classifier: {str(e)}", exc_info=True)
            return create_error_state(state, "m_classifier", str(e), "stage_classifier")
    
    def _validate_classification_result(self, consensus_result: Dict, classification_type: str) -> bool:
        """Validate the classification result based on type."""
        valid_classifications = {
            'T': TNMOutputParser.VALID_T_CLASSIFICATIONS,
            'N': TNMOutputParser.VALID_N_CLASSIFICATIONS,
            'M': TNMOutputParser.VALID_M_CLASSIFICATIONS
        }
        
        if not consensus_result:
            return False
            
        if classification_type == 'TNM':
            t_class = consensus_result.get('t_classification')
            n_class = consensus_result.get('n_classification')
            m_class = consensus_result.get('m_classification')
            
            return (t_class in valid_classifications['T'] and 
                    n_class in valid_classifications['N'] and 
                    m_class in valid_classifications['M'])
        
        return False
    
    def retry_classification(self, agent, input_data, parser, classification_type: str, max_retries: int = 100):
        retry_count = 0
        base_temperature = self.config.get('model_settings', {}).get('openai', {}).get('temperature', 0.0)
        current_temperature = base_temperature
        
        while retry_count < max_retries:
            try:
                # Update temperature in agent's LLM
                if retry_count > 0:
                    if hasattr(agent.agent.llm, 'temperature'):
                        agent.agent.llm.temperature = current_temperature
                        self.logger.info(f"Adjusted temperature to {current_temperature} for attempt {retry_count + 1}")

                # Get classification from agent
                result = agent.invoke(input_data)
                
                try:
                    # Parse the result
                    if isinstance(result, dict) and 'output' in result:
                        text_result = result['output']
                    else:
                        text_result = str(result)
                    
                    # Let the parser handle all validation
                    parsed_result = parser.parse(text_result)
                    
                    # If we get here, the format is valid
                    self.logger.info(
                        f"Successfully got valid {classification_type} classification "
                        f"after {retry_count + 1} attempts with temperature {current_temperature}"
                    )
                    return parsed_result
                        
                except ValueError as parse_error:
                    # Increase temperature for next attempt
                    current_temperature = min(1.0, current_temperature + 0.1)
                    
                    self.logger.warning(
                        f"Attempt {retry_count + 1}/{max_retries}: Invalid TNM format: {str(parse_error)}\n"
                        f"Raw response: {text_result}\nIncreasing temperature to {current_temperature}"
                    )
                    
                    # Add error feedback to prompt
                    original_prompt = input_data.get('prompt', '')
                    error_feedback = f"\n\nPREVIOUS ERROR: {str(parse_error)}\nPlease provide response ONLY with the exact allowed values."
                    input_data['prompt'] = original_prompt + error_feedback
                
            except Exception as e:
                current_temperature = min(1.0, current_temperature + 0.1)
                self.logger.warning(
                    f"Attempt {retry_count + 1}/{max_retries}: Error in {classification_type} "
                    f"classification: {str(e)}\nIncreasing temperature to {current_temperature}"
                )
            
            retry_count += 1
        
        self.logger.error(
            f"Failed to get valid {classification_type} classification after "
            f"{max_retries} attempts"
        )
        return None



    def stage_classifier_node(self, state: AgentState) -> AgentState:
        """Stage classification node using rule-based logic"""
        self.logger.debug(f"Stage classifier processing case {state['input'].get('case_number')}")
        try:
            # Get TNM classifications and reasonings from state
            t_classification = state.get('t_classification') or \
                            state['input'].get('t_classification', {}).get('classification')
            n_classification = state.get('n_classification') or \
                            state['input'].get('n_classification', {}).get('classification')
            m_classification = state.get('m_classification') or \
                            state['input'].get('m_classification', {}).get('classification')

            # Get reasonings
            t_reasoning = state['input'].get('t_classification', {}).get('reasoning', '')
            n_reasoning = state['input'].get('n_classification', {}).get('reasoning', '')
            m_reasoning = state['input'].get('m_classification', {}).get('reasoning', '')

            if not all([t_classification, n_classification, m_classification]):
                raise ValueError("Missing TNM classifications")

            # Log the classifications being used
            self.logger.info(f"Using classifications - T: {t_classification}, N: {n_classification}, M: {m_classification}")

            # Determine stage
            # Check for AJCC version-specific stage_rules first, then fallback to global stage_rules
            ajcc_edition = self.config.get('ajcc_edition', 'ajcc8th')
            if ajcc_edition == 'ajcc9th':
                ajcc9th_prompts = self.config.get('ajcc9th_prompts', {})
                stage_rules = ajcc9th_prompts.get('stage_rules', {})
                # Fallback to global stage_rules if not in ajcc9th_prompts
                if not stage_rules:
                    stage_rules = self.config.get('stage_rules', {})
            else:
                stage_rules = self.config.get('stage_rules', {})
            
            stage = determine_stage_from_tnm(
                t_classification, n_classification, m_classification, stage_rules
            )
            
            # Generate stage reasoning - more concise format
            stage_reasoning = (
                f"Stage {stage} determined based on TNM classification: "
                f"T: {t_classification}, N: {n_classification}, M: {m_classification}. "
                f"Stage calculation follows AJCC staging rules."
            )
            
            # Create stage classification result
            # Stage is rule-based (not LLM-generated), so no raw_output exists
            stage_classification = {
                "classification": stage,
                "reasoning": stage_reasoning,
                "raw_output": "Rule-based calculation (no LLM response)"
            }

            # Update state
            state['input']['stage_classification'] = stage_classification

            result_state = {
                **state,
                "next": "final_save",
                "stage_classification": stage
            }

            self.logger.info(f"Determined stage {stage} for case {state['input'].get('case_number')}")
            return result_state

        except Exception as e:
            self.logger.error(f"Error in Stage Classifier: {str(e)}", exc_info=True)
            return create_error_state(state, "stage_classifier", str(e), "final_save")

        


    def run(self, initial_state: Dict) -> Optional[Dict]:
        try:
            # Store True TNM data separately (exclude from model input)
            true_tnm = initial_state.pop("true_tnm", {})
            self.logger.debug(f"Initial true TNM data: {true_tnm}")
            self._true_tnm_data = true_tnm  # Store as class instance variable
            
            # Prepare state for model input (exclude true_tnm)
            model_state = {
                "input": initial_state["input"],
                "iteration_count": initial_state.get("iteration_count", 0)
            }
            
            graph = self.workflow.compile()
            final_state = None
            
            for state in graph.stream(model_state):
                current_step = state.get("next")
                if current_step:
                    self.logger.info(f"Current step: {current_step}")
                
                if current_step == "final_save":
                    # Add true_tnm data at final stage
                    state = dict(state)
                    state["true_tnm"] = self._true_tnm_data
                    self.logger.debug(f"Added true TNM data to final state: {self._true_tnm_data}")
                    
                final_state = state

            if final_state:
                self.logger.info("Workflow completed successfully")
            else:
                self.logger.warning("Workflow did not complete")
            
            return final_state
                
        except Exception as e:
            self.logger.error(f"Error running workflow: {e}", exc_info=True)
            raise

    def final_save_node(self, state: AgentState) -> AgentState:
        self.logger.debug(f"Entering Final save node. Initial state: {state}")
        try:
            # Get True TNM data from class instance variable
            true_tnm = getattr(self, '_true_tnm_data', {})
            self.logger.debug(f"Retrieved true TNM data: {true_tnm}")
            
            # Get classifications (can be string or dict)
            t_class = state.get('t_classification') or state['input'].get('t_classification', {}).get('classification', '')
            n_class = state.get('n_classification') or state['input'].get('n_classification', {}).get('classification', '')
            m_class = state.get('m_classification') or state['input'].get('m_classification', {}).get('classification', '')
            stage_class = state.get('stage_classification') or state['input'].get('stage_classification', {}).get('classification', '')
            
            # Get reasonings from state
            t_reasoning = state['input'].get('t_classification', {}).get('reasoning', '')
            n_reasoning = state['input'].get('n_classification', {}).get('reasoning', '')
            m_reasoning = state['input'].get('m_classification', {}).get('reasoning', '')
            stage_reasoning = state['input'].get('stage_classification', {}).get('reasoning', '')
            
            # Log reasoning extraction for debugging
            self.logger.info(f"Final save - T reasoning length: {len(t_reasoning)}, N reasoning length: {len(n_reasoning)}, M reasoning length: {len(m_reasoning)}")
            self.logger.debug(f"Final save - T reasoning: {t_reasoning[:200] if t_reasoning else 'EMPTY'}")
            self.logger.debug(f"Final save - N reasoning: {n_reasoning[:200] if n_reasoning else 'EMPTY'}")
            self.logger.debug(f"Final save - M reasoning: {m_reasoning[:200] if m_reasoning else 'EMPTY'}")
            
            # Prepare final classification data
            final_classification = {
                'input': state['input'],
                'case_number': state['input'].get('case_number'),
                'histology_category': state.get('histology_category'),
                'histology_subcategory': state.get('histology_subcategory', ''),
                'histology_type': state.get('histology_type'),
                'histology_confidence': state.get('histology_confidence'),
                'histology_reason': state.get('histology_reason'),
                'classifications': {
                    't_classification': t_class,
                    'n_classification': n_class,
                    'm_classification': m_class,
                    'stage_classification': stage_class
                },
                'reasonings': {
                    't_reasoning': t_reasoning,
                    'n_reasoning': n_reasoning,
                    'm_reasoning': m_reasoning,
                    'stage_reasoning': stage_reasoning
                },
                'true_tnm': true_tnm
            }
            
            # Save to file
            output_file = self.config.get('output_file')
            json_file_path = self.config.get('json_output_file')
            save_to_csv_file(final_classification, output_file, self.fieldnames)
            save_to_json_file(final_classification, state['input'], json_file_path)
            
            result_state = {
                **state,
                "next": None,
                "final_classification": final_classification
            }
            
            self.logger.debug(f"Final classification completed for case {state['input'].get('case_number')}")
            return result_state
            
        except Exception as e:
            self.logger.error(f"Error in Final Save: {e}", exc_info=True)
            return create_error_state(state, "final_save", str(e), None)
        
    def finalize_csv(self):
        """Finalize CSV file creation"""
        try:
            output_file = self.config.get('output_file')
            finalize_csv(output_file, self.fieldnames)
        finally:
            # Clean up temp file if it exists
            if hasattr(self, 'temp_file') and self.temp_file:
                try:
                    self.temp_file.close()
                    if os.path.exists(self.temp_file_path):
                        os.remove(self.temp_file_path)
                except Exception as e:
                    self.logger.error(f"Error cleaning up temporary file: {e}")

def main():
    config = Config()
    logger = setup_logging(config)
    
    workflow = TNMClassificationWorkflow(config)
    
    logger.info("Processing Excel data")
    processed_data = process_excel_data(config)
    logger.info(f"Processed {len(processed_data)} cases from Excel")

    success_count = 0
    incomplete_count = 0
    error_count = 0

    try:
        for case in processed_data:
            case_number = case['case_number']
            logger.info(f"Processing case {case_number} ({case_number}/{len(processed_data)})")
            
            initial_state = {
                "input": case['input'],
                "true_tnm": case['true_tnm'],  # Include true TNM data
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
                logger.warning(f"Case {case_number} processing ended at unexpected state: {final_state.get('next')}")
                incomplete_count += 1
        
        logger.info("Finalizing CSV file")
        workflow.finalize_csv()
        logger.info("All cases processed and final CSV file created")
        
        # Calculate and log comprehensive metrics
        accuracy_metrics, confusion_metrics, detailed_results = calculate_metrics(workflow)
        
        if accuracy_metrics:
            logger.info("\n=== Classification Results Summary ===")
            
            # Print overall accuracy summary
            logger.info("\nOverall Performance:")
            for category in ['T', 'N', 'M', 'Stage']:
                accuracy = accuracy_metrics.get(category, {})
                if accuracy:
                    logger.info(f"{category:<6} Accuracy: {accuracy.get('accuracy', 0):.2f}% "
                              f"({accuracy.get('correct', 0)}/{accuracy.get('total', 0)})")
            
            # Print detailed analysis for each category
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
            logger.warning("No accuracy metrics available")
            
            # Print case-by-case details if available
            if detailed_results:
                logger.info("\nCase-by-Case Analysis:")
                for case in detailed_results:
                    logger.info(f"\nCase {case['case_id']} (Hospital #: {case['hospital_number']}):")
                    for category, comparison in case['comparisons'].items():
                        result = "✓" if comparison['correct'] else "✗"
                        logger.info(f"  {category:<6}: {comparison['true']} -> {comparison['predicted']} {result}")
        
        logger.info(f"\nProcessing Summary:")
        logger.info(f"Total Cases: {len(processed_data)}")
        logger.info(f"Successfully Processed: {success_count}")
        logger.info(f"Incomplete Processing: {incomplete_count}")
        logger.info(f"Processing Errors: {error_count}")
        
    except Exception as e:
        logger.critical(f"Critical error in main workflow: {str(e)}", exc_info=True)
    finally:
        if hasattr(workflow, 'temp_file'):
            logger.debug("Closing temporary file")
            workflow.temp_file.close()
    
    logger.info("TNM Classification Workflow completed")

if __name__ == "__main__":
    main()
