"""Google AI Embeddings ModelClient integration — google-genai SDK."""

import os
import logging
import backoff
from typing import Dict, Any, Optional, List, Sequence

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput

from google import genai as google_genai

log = logging.getLogger(__name__)


class GoogleEmbedderClient(ModelClient):
    __doc__ = r"""A component wrapper for Google AI Embeddings API client.

    This client provides access to Google's embedding models through the Google AI API.
    It supports text embeddings for various tasks including semantic similarity,
    retrieval, and classification.

    Args:
        api_key (Optional[str]): Google AI API key. Defaults to None.
            If not provided, will use the GOOGLE_API_KEY environment variable.
        env_api_key_name (str): Environment variable name for the API key.
            Defaults to "GOOGLE_API_KEY".

    Example:
        ```python
        from api.google_embedder_client import GoogleEmbedderClient
        import adalflow as adal

        client = GoogleEmbedderClient()
        embedder = adal.Embedder(
            model_client=client,
            model_kwargs={
                "model": "gemini-embedding-001",
                "task_type": "SEMANTIC_SIMILARITY"
            }
        )
        ```

    References:
        - Google AI Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
        - Available models: gemini-embedding-001
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        env_api_key_name: str = "GOOGLE_API_KEY",
    ):
        """Initialize Google AI Embeddings client.
        
        Args:
            api_key: Google AI API key. If not provided, uses environment variable.
            env_api_key_name: Name of environment variable containing API key.
        """
        super().__init__()
        self._api_key = api_key or os.getenv(env_api_key_name)
        if not self._api_key:
            raise ValueError(
                f"Environment variable {env_api_key_name} must be set"
            )
        self._client = google_genai.Client(api_key=self._api_key)

    def parse_embedding_response(self, response) -> EmbedderOutput:
        """Parse Google AI embedding response to EmbedderOutput format.
        
        Args:
            response: google.genai.types.EmbedContentResponse
            
        Returns:
            EmbedderOutput with parsed embeddings
        """
        try:
            from adalflow.core.types import Embedding
            
            embedding_data = []

            def _extract_embedding_value(obj):
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    if "embedding" in obj:
                        return obj.get("embedding")
                    if "embeddings" in obj:
                        return obj.get("embeddings")
                if hasattr(obj, "embedding"):
                    return getattr(obj, "embedding")
                if hasattr(obj, "embeddings"):
                    return getattr(obj, "embeddings")
                for method_name in ("model_dump", "to_dict", "dict"):
                    if hasattr(obj, method_name):
                        try:
                            dumped = getattr(obj, method_name)()
                            if isinstance(dumped, dict):
                                if "embedding" in dumped:
                                    return dumped.get("embedding")
                                if "embeddings" in dumped:
                                    return dumped.get("embeddings")
                        except Exception:
                            pass
                return None
            
            embedding_value = _extract_embedding_value(response)
            if embedding_value is None:
                log.warning("Unexpected embedding response type/structure: %s", type(response))
                embedding_data = []
            elif isinstance(embedding_value, list) and len(embedding_value) > 0:
                if isinstance(embedding_value[0], (int, float)):
                    embedding_data = [Embedding(embedding=embedding_value, index=0)]
                elif isinstance(embedding_value[0], list):
                    embedding_data = [
                        Embedding(embedding=emb_list, index=i)
                        for i, emb_list in enumerate(embedding_value)
                        if isinstance(emb_list, list) and len(emb_list) > 0
                    ]
                else:
                    extracted = []
                    for item in embedding_value:
                        item_emb = _extract_embedding_value(item)
                        if isinstance(item_emb, list) and len(item_emb) > 0:
                            extracted.append(item_emb)
                    embedding_data = [
                        Embedding(embedding=emb_list, index=i)
                        for i, emb_list in enumerate(extracted)
                    ]
            else:
                log.warning("Empty or invalid embedding data parsed from response")
                embedding_data = []

            if embedding_data:
                first_dim = len(embedding_data[0].embedding) if embedding_data[0].embedding is not None else 0
                log.info("Parsed %s embedding(s) (dim=%s)", len(embedding_data), first_dim)
            
            return EmbedderOutput(
                data=embedding_data,
                error=None,
                raw_response=response
            )
        except Exception as e:
            log.error(f"Error parsing Google AI embedding response: {e}")
            return EmbedderOutput(
                data=[],
                error=str(e),
                raw_response=response
            )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to Google AI API format.
        
        Args:
            input: Text input(s) to embed
            model_kwargs: Model parameters including model name and task_type
            model_type: Should be ModelType.EMBEDDER for this client
            
        Returns:
            Dict: API kwargs for Google AI embedding call
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError(f"GoogleEmbedderClient only supports EMBEDDER model type, got {model_type}")
        
        # Ensure input is a list
        if isinstance(input, str):
            content = [input]
        elif isinstance(input, Sequence):
            content = list(input)
        else:
            raise TypeError("input must be a string or sequence of strings")
        
        final_model_kwargs = model_kwargs.copy()
        
        # Handle single vs batch embedding (both use "contents")
        final_model_kwargs["contents"] = content
            
        # Set default task type if not provided
        if "task_type" not in final_model_kwargs:
            final_model_kwargs["task_type"] = "SEMANTIC_SIMILARITY"
            
        # Set default model if not provided
        if "model" not in final_model_kwargs:
            final_model_kwargs["model"] = "gemini-embedding-001"
            
        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Google AI may raise various exceptions
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call Google AI embedding API.
        
        Args:
            api_kwargs: API parameters
            model_type: Should be ModelType.EMBEDDER
            
        Returns:
            Google AI embedding response
        """
        if model_type != ModelType.EMBEDDER:
            raise ValueError(f"GoogleEmbedderClient only supports EMBEDDER model type")
            
        safe_log_kwargs = {k: v for k, v in api_kwargs.items() if k not in {"contents"}}
        if "contents" in api_kwargs:
            try:
                contents = api_kwargs.get("contents")
                safe_log_kwargs["contents_count"] = len(contents) if hasattr(contents, "__len__") else None
            except Exception:
                safe_log_kwargs["contents_count"] = None
        log.info("Google AI Embeddings call kwargs (sanitized): %s", safe_log_kwargs)
        
        try:
            model = api_kwargs.pop("model", "gemini-embedding-001")
            contents = api_kwargs.pop("contents")
            task_type = api_kwargs.pop("task_type", "SEMANTIC_SIMILARITY")
            
            response = self._client.models.embed_content(
                model=model,
                contents=contents,
                config={"task_type": task_type, **api_kwargs},
            )
            return response
            
        except Exception as e:
            log.error(f"Error calling Google AI Embeddings API: {e}")
            raise

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Async call to Google AI embedding API.
        
        Note: google-genai client supports async via aio.
        """
        try:
            model = api_kwargs.pop("model", "gemini-embedding-001")
            contents = api_kwargs.pop("contents")
            task_type = api_kwargs.pop("task_type", "SEMANTIC_SIMILARITY")
            
            response = await self._client.aio.models.embed_content(
                model=model,
                contents=contents,
                config={"task_type": task_type, **api_kwargs},
            )
            return response
        except Exception as e:
            log.error(f"Error calling Google AI Embeddings API (async): {e}")
            raise