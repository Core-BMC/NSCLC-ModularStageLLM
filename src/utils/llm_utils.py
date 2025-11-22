"""LLM setup and configuration utilities."""

import logging
import os
import random
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

logger = logging.getLogger(__name__)


def setup_llm(config: Dict[str, Any]) -> ChatOpenAI:
    """Setup default LLM based on configuration.

    Args:
        config: Configuration dictionary with model_settings

    Returns:
        Configured LLM instance

    Raises:
        Exception: If LLM setup fails
    """
    try:
        # Get settings from model_settings
        model_settings = config.get('model_settings', {})
        llm_choice = model_settings.get('llm_choice', 'local')

        if llm_choice == "local":
            # Load environment variables
            load_dotenv()

            # Use local API settings
            local_settings = model_settings.get('local', {})

            # Allow environment variables to override YAML settings
            base_url = (
                os.getenv('LOCAL_API_BASE_URL') or
                local_settings.get('base_url')
            )
            api_key = (
                os.getenv('LOCAL_API_KEY') or
                local_settings.get('api_key')
            )
            model_name = (
                os.getenv('LOCAL_API_MODEL_NAME') or
                local_settings.get('name')
            )
            temperature = float(
                os.getenv('LOCAL_API_TEMPERATURE') or
                local_settings.get('temperature', 0.0)
            )
            
            # Check if SSL verification should be disabled (for self-signed certificates)
            verify_ssl = local_settings.get('verify_ssl', True)
            if isinstance(verify_ssl, str):
                verify_ssl = verify_ssl.lower() in ('true', '1', 'yes')
            
            # Ensure base_url ends with /v1 if it doesn't already
            # ChatOpenAI automatically appends /chat/completions, so base_url should be like https://host:port/v1
            if base_url and not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
                # Check if /v1 is already in the path
                if '/v1/' not in base_url:
                    base_url = base_url.rstrip('/') + '/v1'
            
            # Create httpx client with SSL verification setting
            http_client = httpx.Client(verify=verify_ssl) if not verify_ssl else None

            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                http_client=http_client
            )
            logger.info(
                f"Using local API with model: {model_name} "
                f"(SSL verification: {verify_ssl})"
            )
            return llm

        elif llm_choice == "azure":
            # Load environment variables
            load_dotenv()
            os.environ["OPENAI_API_TYPE"] = "azure"

            # Use Azure settings
            azure_settings = model_settings.get('azure', {})
            
            # Handle temperature: use temperature if set, otherwise use average of temperature_low and temperature_high
            if 'temperature' in azure_settings:
                temperature = float(azure_settings.get('temperature', 0.0))
            else:
                temp_low = azure_settings.get('temperature_low', 0.0)
                temp_high = azure_settings.get('temperature_high', 0.0)
                temperature = (float(temp_low) + float(temp_high)) / 2.0
            
            llm = AzureChatOpenAI(
                azure_deployment=azure_settings.get('name', 'gpt-4o'),
                temperature=temperature,
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            logger.info(
                f"Using Azure OpenAI API with model: {azure_settings.get('name')} "
                f"(temperature: {temperature})"
            )
            return llm

        else:  # openai
            # Load environment variables
            load_dotenv()

            # Use OpenAI settings
            openai_settings = model_settings.get('openai', {})
            model_name = openai_settings.get('name', 'gpt-4o')
            
            # Handle temperature: use temperature if set, otherwise use average of temperature_low and temperature_high
            if 'temperature' in openai_settings:
                temperature = float(openai_settings.get('temperature', 0.0))
            else:
                temp_low = openai_settings.get('temperature_low', 0.0)
                temp_high = openai_settings.get('temperature_high', 0.0)
                temperature = (float(temp_low) + float(temp_high)) / 2.0

            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info(
                f"Using OpenAI API with model: {model_name} "
                f"(temperature: {temperature})"
            )
            return llm

    except Exception as e:
        logger.error(f"Error setting up LLMs: {e}")
        raise


def get_llm_instance(config: Dict[str, Any]) -> ChatOpenAI:
    """Get LLM instance with random temperature.

    Args:
        config: Configuration dictionary with model_settings

    Returns:
        LLM instance with random temperature

    Raises:
        Exception: If LLM creation fails
    """
    try:
        model_settings = config.get('model_settings', {})
        llm_choice = model_settings.get('llm_choice', 'local')

        if llm_choice == "local":
            local_settings = model_settings.get('local', {})
            temperature = random.uniform(
                local_settings.get('temperature_low', 0.0),
                local_settings.get('temperature_high', 1.0)
            )
            
            # Check if SSL verification should be disabled (for self-signed certificates)
            verify_ssl = local_settings.get('verify_ssl', True)
            if isinstance(verify_ssl, str):
                verify_ssl = verify_ssl.lower() in ('true', '1', 'yes')
            
            base_url = os.getenv('LOCAL_API_BASE_URL') or local_settings.get('base_url')
            
            # Ensure base_url ends with /v1 if it doesn't already
            # ChatOpenAI automatically appends /chat/completions, so base_url should be like https://host:port/v1
            if base_url and not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
                # Check if /v1 is already in the path
                if '/v1/' not in base_url:
                    base_url = base_url.rstrip('/') + '/v1'
            
            # Create httpx client with SSL verification setting
            http_client = httpx.Client(verify=verify_ssl) if not verify_ssl else None
            
            return ChatOpenAI(
                base_url=base_url,
                api_key=os.getenv('LOCAL_API_KEY') or
                local_settings.get('api_key'),
                model_name=os.getenv('LOCAL_API_MODEL_NAME') or
                local_settings.get('name'),
                temperature=temperature,
                http_client=http_client
            )
        elif llm_choice == "azure":
            azure_settings = model_settings.get('azure', {})
            temperature = random.uniform(
                azure_settings.get('temperature_low', 0.0),
                azure_settings.get('temperature_high', 1.0)
            )
            return AzureChatOpenAI(
                azure_deployment=azure_settings.get('name', 'gpt-4o'),
                temperature=temperature,
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        else:  # openai
            openai_settings = model_settings.get('openai', {})
            temperature = random.uniform(
                openai_settings.get('temperature_low', 0.0),
                openai_settings.get('temperature_high', 1.0)
            )
            return ChatOpenAI(
                model_name=openai_settings.get('name', 'gpt-4o'),
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
    except Exception as e:
        logger.error(f"Error creating LLM instance: {e}")
        raise

