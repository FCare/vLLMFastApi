"""
Schémas de données pour l'API
"""
from .openai_schemas import *
from .function_schemas import *

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest", 
    "ChatCompletionResponse",
    "ChatCompletionStreamResponse",
    "ModelInfo",
    "ModelsListResponse",
    "ErrorResponse",
    "HealthStatus",
    "MetricsResponse",
    "FunctionCall",
    "FunctionResult",
    "QwenFunctionCall",
    "QwenToolMessage"
]