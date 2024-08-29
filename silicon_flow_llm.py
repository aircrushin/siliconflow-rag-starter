from langchain.llms.base import LLM
from openai import OpenAI
from pydantic import Field
from typing import Any, List, Optional

class SiliconFlowLLM(LLM):
    client: OpenAI = Field(default_factory=lambda: None)
    model_name: str = Field()
    api_key: str = Field()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.siliconflow.cn/v1")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "silicon_flow"

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}