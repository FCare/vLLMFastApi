"""
Schémas compatibles avec l'API OpenAI pour FastAPI
"""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import json


class ChatMessage(BaseModel):
    """Message dans une conversation"""
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Rôle de l'émetteur du message"
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        description="Contenu du message (texte ou multi-modal)"
    )
    name: Optional[str] = Field(None, description="Nom de l'utilisateur ou de l'outil")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Appels d'outils demandés par l'assistant"
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID de l'appel d'outil pour les réponses tool"
    )


class FunctionDefinition(BaseModel):
    """Définition d'une fonction disponible"""
    name: str = Field(description="Nom de la fonction")
    description: Optional[str] = Field(None, description="Description de la fonction")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Schéma JSON des paramètres"
    )


class ToolDefinition(BaseModel):
    """Définition d'un outil (fonction)"""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """Requête de chat completion compatible OpenAI"""
    model: str = Field(description="Modèle à utiliser")
    messages: List[ChatMessage] = Field(description="Messages de la conversation")
    
    # Paramètres de génération
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(1024, ge=1, le=32768)
    
    # Function calling
    tools: Optional[List[ToolDefinition]] = Field(None, description="Outils disponibles")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        "auto", description="Contrôle de l'utilisation des outils"
    )
    
    # Streaming
    stream: Optional[bool] = Field(False, description="Streaming des réponses")
    
    # Paramètres avancés
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1, le=100)
    
    # Métadonnées
    user: Optional[str] = Field(None, description="ID utilisateur pour tracking")


class ChatCompletionChoice(BaseModel):
    """Choix de réponse dans une chat completion"""
    index: int = Field(description="Index du choix")
    message: ChatMessage = Field(description="Message généré")
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = Field(
        description="Raison de fin de génération"
    )


class ChatCompletionUsage(BaseModel):
    """Statistiques d'usage des tokens"""
    prompt_tokens: int = Field(description="Tokens dans le prompt")
    completion_tokens: int = Field(description="Tokens dans la réponse")
    total_tokens: int = Field(description="Total des tokens utilisés")


class ChatCompletionResponse(BaseModel):
    """Réponse de chat completion"""
    id: str = Field(description="ID unique de la requête")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(description="Timestamp de création")
    model: str = Field(description="Modèle utilisé")
    choices: List[ChatCompletionChoice] = Field(description="Choix générés")
    usage: ChatCompletionUsage = Field(description="Statistiques d'usage")
    system_fingerprint: Optional[str] = Field(None, description="Empreinte du système")


class ChatCompletionStreamDelta(BaseModel):
    """Delta pour streaming"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionStreamChoice(BaseModel):
    """Choix pour streaming"""
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Réponse streaming"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    """Information sur un modèle"""
    id: str = Field(description="ID du modèle")
    object: Literal["model"] = "model"
    created: int = Field(description="Timestamp de création")
    owned_by: str = Field(description="Propriétaire du modèle")
    permission: Optional[List[Dict[str, Any]]] = Field(None)


class ModelsListResponse(BaseModel):
    """Liste des modèles disponibles"""
    object: Literal["list"] = "list"
    data: List[ModelInfo] = Field(description="Liste des modèles")


class ErrorDetail(BaseModel):
    """Détail d'une erreur"""
    message: str = Field(description="Message d'erreur")
    type: str = Field(description="Type d'erreur")
    code: Optional[str] = Field(None, description="Code d'erreur")


class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    error: ErrorDetail = Field(description="Détails de l'erreur")


# Schémas pour le health check et monitoring
class HealthStatus(BaseModel):
    """État de santé du service"""
    status: Literal["healthy", "unhealthy", "loading"] = Field(description="État général")
    model_loaded: bool = Field(description="Modèle chargé")
    gpu_available: bool = Field(description="GPU disponible")
    memory_usage: Dict[str, Any] = Field(description="Utilisation mémoire")
    uptime: float = Field(description="Temps de fonctionnement en secondes")
    version: str = Field(description="Version du service")


class MetricsResponse(BaseModel):
    """Métriques du service"""
    requests_total: int = Field(description="Nombre total de requêtes")
    requests_per_minute: float = Field(description="Requêtes par minute")
    average_response_time: float = Field(description="Temps de réponse moyen")
    active_connections: int = Field(description="Connexions actives")
    queue_size: int = Field(description="Taille de la queue")
    gpu_utilization: float = Field(description="Utilisation GPU (%)")
    memory_used: float = Field(description="Mémoire utilisée (GB)")