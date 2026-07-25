from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.trustedrouter.com/v1"
DEFAULT_MODEL = "trustedrouter/zdr"


class TrustedRouter(OpenAILike):
    """
    TrustedRouter LLM.

    TrustedRouter (https://trustedrouter.com) is an OpenAI-compatible LLM router that serves many models behind one endpoint.

    To instantiate the `TrustedRouter` class, you will need to provide an API key. You can set the API key either as an environment variable `TRUSTEDROUTER_API_KEY` or directly in the class
    constructor. If you haven't signed up for an API key yet, you can do so on the TrustedRouter website at (https://trustedrouter.com). Once you have your API key, you can use the `TrustedRouter`
    class to interact with the LLM for tasks like chatting, streaming, and completing prompts.

    Examples:
        `pip install llama-index-llms-trustedrouter`

        ```python
        from llama_index.llms.trustedrouter import TrustedRouter

        llm = TrustedRouter(
            api_key="<your-api-key>",
            max_tokens=256,
            context_window=4096,
            model="trustedrouter/zdr",
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

    model: str = Field(
        description="The TrustedRouter model to use. See https://trustedrouter.com/models for options."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model. See https://trustedrouter.com/models for options.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_base = get_from_param_or_env(
            "api_base", api_base, "TRUSTEDROUTER_API_BASE"
        )
        api_key = get_from_param_or_env("api_key", api_key, "TRUSTEDROUTER_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TrustedRouter_LLM"
