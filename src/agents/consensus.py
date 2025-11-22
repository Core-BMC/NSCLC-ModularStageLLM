"""Consensus mechanism for TNM classification."""

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.data_utils import format_input_data
from src.utils.tnm_config_utils import filter_special_notes

logger = logging.getLogger(__name__)


class AgentConsensus:
    """Consensus mechanism for TNM classification."""

    def __init__(self, max_retries=4):
        """Initialize consensus agent.

        Args:
            max_retries: Maximum number of retries for consensus
        """
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self.MAX_ATTEMPTS = 50

        # Minimum responses required per classification type
        self.MIN_RESPONSES = {
            "T": 1,  # T node requires 1 response
            "N": 1,  # Other nodes require 1 response
            "M": 1
        }

        # Valid classifications per type
        self.VALID_CLASSIFICATIONS = {
            "T": {'T0', 'T1a', 'T1b', 'T1c', 'T2a', 'T2b', 'T3', 'T4'},
            "N": {'N0', 'N1', 'N2', 'N3'},
            "M": {'M0', 'M1a', 'M1b', 'M1c'}
        }

        # Consensus thresholds (100 / number of possible classifications)
        self.CONSENSUS_THRESHOLDS = {
            "T": 100 / 2,
            "N": 100 / 2,
            "M": 100 / 2
        }

    def collect_responses(
        self,
        workflow,
        input_data: Dict[str, Any],
        parser,
        agent_type: str,
        use_consensus: bool = True
    ) -> Optional[Dict]:
        """Collect responses and reach consensus using workflow instance.

        Args:
            workflow: Workflow instance for LLM access
            input_data: Input data dictionary
            parser: Parser instance for classification
            agent_type: Type of agent (T, N, or M)
            use_consensus: If True, collect multiple responses and reach consensus.
                         If False, return single response only.

        Returns:
            Consensus result dictionary or None if failed
        """
        valid_responses = []
        total_attempts = 0

        # Determine minimum responses based on consensus mode
        if use_consensus:
            required_responses = self.MIN_RESPONSES.get(agent_type, 3)
        else:
            # Single response mode: only need 1 response
            required_responses = 1

        self.logger.info(f"\n{'='*50}")
        mode_str = "CONSENSUS MODE" if use_consensus else "SINGLE RESPONSE MODE"
        self.logger.info(f"Starting {mode_str} for {agent_type}")
        self.logger.info(f"Required responses: {required_responses}")

        # Step 1: Collect required number of valid responses
        while (len(valid_responses) < required_responses and
               total_attempts < self.MAX_ATTEMPTS):
            self.logger.info(
                f"\nAttempt {total_attempts + 1}/{self.MAX_ATTEMPTS}"
            )
            response = self._get_single_response(
                workflow, input_data, parser, agent_type
            )
            if response:
                valid_responses.append(response)
                self.logger.info(
                    f"Valid responses collected: "
                    f"{len(valid_responses)}/{required_responses}"
                )
            total_attempts += 1

        # Check minimum response count
        if len(valid_responses) < required_responses:
            self.logger.error(
                f"Failed to collect minimum {required_responses} "
                f"valid responses for {agent_type}. "
                f"Only collected: {len(valid_responses)}"
            )
            return None

        # If single response mode, return the first valid response
        if not use_consensus:
            if valid_responses:
                result = valid_responses[0]
                self.logger.info(
                    f"Single response mode: Returning first valid response "
                    f"for {agent_type}: {result['classification']}"
                )
                return result
            else:
                self.logger.error(f"No valid responses collected for {agent_type}")
                return None

        # Step 2: Try to reach consensus (only in consensus mode)
        while total_attempts < self.MAX_ATTEMPTS:
            consensus = self._try_reach_consensus(valid_responses, agent_type)
            if consensus:
                self.logger.info(
                    f"Consensus reached for {agent_type}: "
                    f"{consensus['classification']}"
                )
                return consensus

            # Collect additional response if consensus failed
            self.logger.info(
                "No consensus reached, collecting additional response"
            )
            response = self._get_single_response(
                workflow, input_data, parser, agent_type
            )
            if response:
                valid_responses.append(response)
                self.logger.info(
                    f"Added new response. Total responses: "
                    f"{len(valid_responses)}"
                )
            total_attempts += 1

        # Step 3: Return majority vote if max attempts reached
        self.logger.warning(
            f"Max attempts ({self.MAX_ATTEMPTS}) reached. Using majority vote."
        )
        return self._get_majority_vote(valid_responses, agent_type)

    def _get_single_response(
        self,
        workflow,
        input_data: Dict[str, Any],
        parser,
        agent_type: str
    ) -> Optional[Dict]:
        """Get and parse single response using workflow instance.

        Args:
            workflow: Workflow instance for LLM access
            input_data: Input data dictionary
            parser: Parser instance for classification
            agent_type: Type of agent (T, N, or M)

        Returns:
            Dictionary with classification, raw_output, and reasoning or None
        """
        try:
            # Get appropriate prompt based on agent type
            agent_type = agent_type.upper()
            if agent_type == "T":
                base_parser = "T0/T1a/T1b/T1c/T2a/T2b/T3/T4"
                base_prompt = workflow.t_classifier_prompt
                if not base_prompt:
                    raise ValueError("T classifier prompt not found")

            elif agent_type == "N":
                base_parser = "N0/N1/N2/N3"
                base_prompt = workflow.n_classifier_prompt
                if not base_prompt:
                    raise ValueError("N classifier prompt not found")

            elif agent_type == "M":
                base_parser = "M0/M1a/M1b/M1c"
                base_prompt = workflow.m_classifier_prompt
                if not base_prompt:
                    raise ValueError("M classifier prompt not found")
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Get special notes
            special_notes = workflow.tnm_criteria.get('special_notes', [])
            filtered_notes = filter_special_notes(
                special_notes, f"{agent_type} category"
            )
            notes_text = "\n".join([f"- {note}" for note in filtered_notes])

            # Format input data
            input_text = format_input_data(
                input_data.get('input', input_data)
            )

            # Use prompt from config file directly (it already contains all necessary instructions including reasoning request)
            # The config prompt already includes:
            # - Expert role description
            # - Classification rules and criteria
            # - JSON format with reasoning field
            task_message = base_prompt

            # Create system and human messages
            system_message = SystemMessage(content=task_message)
            human_message = HumanMessage(
                content=f"Please provide the {agent_type} classification:\n\n"
                f"{input_text}"
            )

            self.logger.debug(f"Using system prompt:\n{task_message}")
            self.logger.info(f"Input data for {agent_type} classifier (first 2000 chars):\n{input_text[:2000]}")
            if len(input_text) > 2000:
                self.logger.info(f"... (truncated, total length: {len(input_text)} chars)")

            # Get response
            llm = workflow._get_llm_instance()
            messages = [system_message, human_message]
            
            self.logger.debug(f"Invoking LLM with {len(messages)} messages")
            try:
                response = llm.invoke(messages)
            except Exception as e:
                self.logger.error(f"LLM invocation failed: {e}", exc_info=True)
                return None

            if not response:
                self.logger.warning("Empty response object from LLM")
                return None
                
            # Handle different response types
            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, str):
                response_content = response
            elif isinstance(response, dict):
                response_content = response.get('content', str(response))
            else:
                response_content = str(response)
            
            if not response_content:
                self.logger.warning("Empty response content from agent")
                return None
            
            self.logger.info(f"LLM response for {agent_type} (full length: {len(response_content)}): {response_content}")
            self.logger.debug(f"LLM response (first 500 chars): {response_content[:500]}")
            # Log full response for debugging reasoning extraction
            if len(response_content) > 500:
                self.logger.debug(f"LLM response (last 500 chars): {response_content[-500:]}")

            # Extract reasoning from response
            reasoning = ''
            
            # Priority 1: Try to extract reasoning from JSON response
            try:
                # Try to find JSON in the response (with code blocks)
                json_pattern = r'```json\s*(\{.*?\})\s*```'
                json_match = re.search(json_pattern, response_content, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                    self.logger.info(f"Found JSON block for {agent_type}: {json_text[:500]}")
                    json_data = json.loads(json_text)
                    if 'reasoning' in json_data:
                        reasoning = json_data['reasoning'].strip()
                        self.logger.info(f"Extracted reasoning from JSON for {agent_type}: {reasoning[:200]}")
                    else:
                        self.logger.warning(f"JSON found for {agent_type} but no 'reasoning' field. JSON keys: {list(json_data.keys())}")
                else:
                    # Try to find JSON without code blocks - improved pattern for nested JSON
                    # Find the first { and then match balanced braces
                    brace_start = response_content.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        json_end = brace_start
                        for i in range(brace_start, len(response_content)):
                            if response_content[i] == '{':
                                brace_count += 1
                            elif response_content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if brace_count == 0:
                            json_text2 = response_content[brace_start:json_end]
                            self.logger.info(f"Found inline JSON for {agent_type}: {json_text2[:500]}")
                            try:
                                json_data = json.loads(json_text2)
                                if 'reasoning' in json_data:
                                    reasoning = json_data['reasoning'].strip()
                                    self.logger.info(f"Extracted reasoning from inline JSON for {agent_type}: {reasoning[:200]}")
                                else:
                                    self.logger.warning(f"Inline JSON found for {agent_type} but no 'reasoning' field. JSON keys: {list(json_data.keys())}")
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Failed to parse inline JSON for {agent_type}: {e}")
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                self.logger.warning(f"Could not extract reasoning from JSON for {agent_type}: {e}")
            
            # Priority 2: Try to extract from <think> tags (used in prompts)
            if not reasoning:
                redacted_pattern = r'<think>(.*?)</think>'
                redacted_match = re.search(redacted_pattern, response_content, re.DOTALL)
                if redacted_match:
                    reasoning = redacted_match.group(1).strip()
                    # Clean up the reasoning text
                    reasoning = re.sub(r'\n+', ' ', reasoning)
                    reasoning = re.sub(r'\s+', ' ', reasoning)
                    self.logger.debug(f"Extracted reasoning from redacted_reasoning tags: {reasoning[:200]}")
            
            # Priority 3: Try to extract from <think> tags (alternative format)
            if not reasoning:
                think_pattern = r'<think>(.*?)</think>'
                think_match = re.search(think_pattern, response_content, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    # Clean up the reasoning text
                    reasoning = re.sub(r'\n+', ' ', reasoning)
                    reasoning = re.sub(r'\s+', ' ', reasoning)
                    self.logger.debug(f"Extracted reasoning from think tags: {reasoning[:200]}")
            
            # Priority 4: Use cleaned full response without JSON as fallback
            if not reasoning:
                # If no tags or JSON reasoning, use cleaned full response without JSON
                # First, try to extract any text before JSON block
                before_json_match = re.search(r'^(.*?)```json', response_content, re.DOTALL)
                if before_json_match:
                    reasoning = before_json_match.group(1).strip()
                    if reasoning:
                        self.logger.info(f"Using text before JSON as reasoning for {agent_type}: {reasoning[:200]}")
                
                # If still no reasoning, use cleaned full response
                if not reasoning:
                    response_text = re.sub(
                        r'```(?:json)?\s*\{[^}]+\}\s*```',
                        '',
                        response_content,
                        flags=re.DOTALL
                    )
                    # Also remove think tags if present
                    response_text = re.sub(
                        r'<think>.*?</think>',
                        '',
                        response_text,
                        flags=re.DOTALL
                    )
                    response_text = re.sub(
                        r'<think>.*?</think>',
                        '',
                        response_text,
                        flags=re.DOTALL
                    )
                    reasoning = response_text.strip()
                    if reasoning:
                        self.logger.info(f"Using cleaned full response as reasoning for {agent_type}: {reasoning[:200]}")
                    # Note: Don't set default reasoning here - will be set after parsing

            self.logger.info(f"Final reasoning extracted for {agent_type} (length: {len(reasoning)}, first 200 chars): {reasoning[:200] if reasoning else 'EMPTY'}")
            self.logger.debug(f"Attempting to parse response with parser: {type(parser).__name__}")
            parsed_result = parser.parse_base(response_content)
            if parsed_result:
                classification = parsed_result['classification']
                self.logger.info(
                    f"Valid Response: {classification}"
                )
                
                # If reasoning is still empty after all extraction attempts, generate a default one
                if not reasoning:
                    # Generate a basic reasoning based on classification
                    reasoning = f"{agent_type} classification {classification} determined based on medical reports and TNM staging criteria."
                    self.logger.info(f"Generated default reasoning for {agent_type}: {reasoning}")
                
                result_dict = {
                    'classification': classification,
                    'raw_output': response_content,
                    'reasoning': reasoning
                }
                self.logger.info(f"Returning result for {agent_type} - classification: {result_dict['classification']}, reasoning length: {len(reasoning)}, reasoning: {reasoning[:100] if reasoning else 'EMPTY'}")
                return result_dict
            else:
                self.logger.warning(f"Parser returned None for {agent_type} response")

        except Exception as e:
            self.logger.warning(f"Response Error: {str(e)}")

        return None

    def _try_reach_consensus(self, responses, agent_type=None):
        """Try to reach consensus from current responses.

        Args:
            responses: List of response dictionaries
            agent_type: Type of agent (T, N, or M)

        Returns:
            Consensus result dictionary or None if no consensus
        """
        if not responses:
            return None

        classification_counts = {}
        response_details = {}

        for resp in responses:
            classification = resp['classification']
            if classification in classification_counts:
                classification_counts[classification] += 1
            else:
                classification_counts[classification] = 1
                # Store the first response for each classification
                response_details[classification] = resp

        total_responses = len(responses)

        # Calculate ratios and find consensus
        classification_ratios = {
            class_: (count / total_responses) * 100
            for class_, count in classification_counts.items()
        }

        # Get consensus threshold
        threshold = self.CONSENSUS_THRESHOLDS.get(agent_type, 50)

        # Find classifications meeting threshold
        consensus_candidates = [
            (class_, ratio)
            for class_, ratio in classification_ratios.items()
            if ratio >= threshold
        ]

        if consensus_candidates:
            # Get the classification with highest ratio
            consensus = max(consensus_candidates, key=lambda x: x[1])[0]
            ratio = classification_ratios[consensus]
            votes = classification_counts[consensus]

            self.logger.info(
                f"Consensus reached for {agent_type}: {consensus} "
                f"({votes}/{total_responses} votes, {ratio:.1f}% > "
                f"threshold {threshold:.1f}%)"
            )

            # Return the stored response details for the consensus classification
            return response_details[consensus]

        self.logger.debug(
            f"No consensus yet for {agent_type}. "
            f"Distribution: {classification_counts}"
        )
        return None

    def _get_majority_vote(self, responses, agent_type):
        """Get majority vote when max attempts reached.

        Args:
            responses: List of response dictionaries
            agent_type: Type of agent (T, N, or M)

        Returns:
            Majority vote result dictionary or None if no responses
        """
        if not responses:
            return None

        classification_counts = {}
        response_details = {}

        for resp in responses:
            classification = resp['classification']
            if classification in classification_counts:
                classification_counts[classification] += 1
            else:
                classification_counts[classification] = 1
                response_details[classification] = resp

        total_responses = len(responses)

        # Calculate ratios for each classification
        classification_ratios = {
            class_: (count / total_responses) * 100
            for class_, count in classification_counts.items()
        }

        # Check threshold
        threshold = self.CONSENSUS_THRESHOLDS.get(agent_type, 50)
        consensus_candidates = [
            (class_, ratio)
            for class_, ratio in classification_ratios.items()
            if ratio >= threshold
        ]

        if consensus_candidates:
            # Select classification with highest ratio
            selected = max(consensus_candidates, key=lambda x: x[1])[0]
            count = classification_counts[selected]
            ratio = classification_ratios[selected]

            self.logger.warning(
                f"Using majority vote for {agent_type}: {selected} "
                f"({count}/{total_responses} votes, {ratio:.1f}% > "
                f"threshold {threshold:.1f}%)"
            )
            return response_details[selected]

        # If no classification meets threshold, select most common
        most_common = max(
            classification_counts.items(), key=lambda x: x[1]
        )[0]
        count = classification_counts[most_common]
        ratio = (count / total_responses) * 100

        self.logger.warning(
            f"No classification met threshold for {agent_type}. "
            f"Using most common: {most_common} "
            f"({count}/{total_responses} votes, {ratio:.1f}% < "
            f"threshold {threshold:.1f}%)"
        )

        return response_details[most_common]

