from llama_index.embeddings.ollama import OllamaEmbedding
from typing import List, Union, Optional

class OllamaEmbeddingWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedder = OllamaEmbedding(model_name=model_name)

    async def create(
        self,
        *,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        extra_query: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        timeout: Optional[float] = None,
    ):
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]

        # Get embeddings asynchronously
        embeddings = await self._get_embeddings(input)

        # Wrap the embeddings in an object with a 'data' attribute
        class EmbeddingResponse:
            def __init__(self, embeddings):
                self.data = [{'embedding': emb} for emb in embeddings]

        return EmbeddingResponse(embeddings)

    async def _get_embeddings(self, input: List[str]):
        # Fetch embeddings asynchronously
        return [self.embedder.get_text_embedding(text) for text in input]