from dataclasses import dataclass

from openai import OpenAI

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_model
        
        kwargs = {}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
            
        self.client = OpenAI(**kwargs)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Connect OpenAI provider with token usage and cost tracking.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None

        cost_usd = None
        if (
            input_tokens is not None
            and output_tokens is not None
            and "gpt-4o-mini" in self.model
        ):
            cost_usd = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

