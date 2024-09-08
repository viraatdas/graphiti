import json
import logging
import typing

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

from graphiti_core.llm_client.ollama_wrapper import OllamaEmbeddingWrapper

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig

logger = logging.getLogger(__name__)

# DEFAULT_MODEL = 'llama-3.1-70b-versatile' 
DEFAULT_MODEL = 'llama-3.1-8b-instant'



class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig()
        super().__init__(config, cache)
        self.client = AsyncGroq(api_key=config.api_key)
        self.model = DEFAULT_MODEL


    def get_embedder(self) -> typing.Any:
           """Get embeddings using the OllamaEmbedding model."""
           return OllamaEmbeddingWrapper(model_name="llama3")

    async def create_embedding(self, input: str) -> list[float]:
        """Create embedding using the OllamaEmbedding model."""
        embedding = self.embed_model.get_text_embedding(input)  # Use the correct method
        return embedding

    async def _generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
        """Generate response using the Groq client."""
        msgs: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
