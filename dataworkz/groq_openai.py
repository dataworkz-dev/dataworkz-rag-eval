import os
from typing import Dict, Optional

from continuous_eval.llms.base import LLMInterface, LLMInterfaceFactory

try:
    from openai import OpenAI as _OpenAI
    GROQ_OPENAI_AVAILABLE = True
except ImportError:
    GROQ_OPENAI_AVAILABLE = False


class GroqOpenAI(LLMInterface):
    """
    Groq OpenAI LLM provider.

    Example:
    ```
    llm = GroqOpenAI(
        api_key="1234567890123",
        endpoint="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile"
    )
    ```

    To register a new provider, use the following:
    ```
    LLMFactory.register_provider(
        "groq_openai",
        model="llama3-8b-8192",
        provider_class=GroqOpenAIFactory(
            api_key="1234567890123",
            endpoint="https://api.groq.com/openai/v1"
        ),
    )
    llm = LLMFactory.get("groq_openai:llama3-8b-8192"
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = "https://api.groq.com/openai/v1",
        model: Optional[str] = "llama-3.3-70b-versatile",
        **kwargs,
    ):
        if not GROQ_OPENAI_AVAILABLE:
            raise ValueError("Groq OpenAI is not available")
        if os.getenv("GROQ_API_KEY") is None and api_key is None:
            raise ValueError(
                "Please set the environment variable GROQ_API_KEY. "
                "You can get one at https://groq.com."
            )
        if os.getenv("GROQ_BASE_URL") is None and endpoint is None:
            raise ValueError(
                "Please set the environment variable GROQ_BASE_URL. "
                "You can get one at https://groq.com."
            )
        self.client = _OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url=endpoint,
        )
        self.defaults = {
            "seed": 0,
            "temperature": 0.001,
            "max_tokens": 4096,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "model": model or "llama-3.3-70b-versatile"
        }
        self.defaults.update(kwargs)

    def run(self, prompt: Dict[str, str], temperature: float = 0) -> str:
        kwargs = self.defaults.copy()
        kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


class GroqOpenAIFactory(LLMInterfaceFactory):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = "https://api.groq.com/openai/v1",
        **kwargs,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.endpoint = endpoint 
        self.extra_kwargs = kwargs

    def __call__(self, model, **kwargs):
        all_kwargs = {**self.extra_kwargs, **kwargs}
        return GroqOpenAI(
            api_key=self.api_key,
            endpoint=self.endpoint,
            **all_kwargs,
        )