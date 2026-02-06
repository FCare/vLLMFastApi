"""
Router pour les endpoints de chat completion compatibles OpenAI
"""
import asyncio
import json
import time
import uuid
import logging
from typing import AsyncGenerator, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse

from ..schemas.openai_schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatMessage,
    ErrorResponse
)
from ..schemas.function_schemas import FunctionRegistry, FunctionCall, EXAMPLE_FUNCTIONS
from ..models.qwen_model import model_manager
from ..utils.async_queue import queue_manager

logger = logging.getLogger(__name__)

# Création du router
router = APIRouter(tags=["chat"])

# Registry des fonctions pour le function calling
function_registry = FunctionRegistry()

# Chargement des fonctions d'exemple
async def _load_example_functions():
    """Charge les fonctions d'exemple"""
    
    async def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
        """Fonction d'exemple pour la météo"""
        # Simulation d'un appel d'API météo
        await asyncio.sleep(0.1)  # Simule latence réseau
        
        temp = 22 if unit == "celsius" else 72
        return {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": "sunny",
            "humidity": 65
        }
    
    async def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
        """Fonction d'exemple pour recherche web"""
        await asyncio.sleep(0.2)  # Simule latence réseau
        
        return {
            "query": query,
            "results": [
                {
                    "title": f"Résultat {i+1} pour '{query}'",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"Description du résultat {i+1}..."
                }
                for i in range(min(num_results, 5))
            ],
            "total_results": num_results
        }
    
    async def analyze_image(image_url: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Fonction d'exemple pour analyse d'image"""
        await asyncio.sleep(0.3)  # Simule traitement image
        
        return {
            "image_url": image_url,
            "analysis_type": analysis_type,
            "detected_objects": ["person", "car", "building"] if analysis_type == "objects" else [],
            "text_content": "Some text found" if analysis_type == "text" else "",
            "description": f"Image analysée avec le type '{analysis_type}'"
        }
    
    # Enregistrement des fonctions
    for func_def in EXAMPLE_FUNCTIONS:
        func_name = func_def["function"]["name"]
        if func_name == "get_current_weather":
            function_registry.register_function(func_def, get_current_weather)
        elif func_name == "search_web":
            function_registry.register_function(func_def, search_web)
        elif func_name == "analyze_image":
            function_registry.register_function(func_def, analyze_image)


# Chargement des fonctions au démarrage
asyncio.create_task(_load_example_functions())


async def _validate_request(request: ChatCompletionRequest) -> None:
    """Valide une requête de chat completion"""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Le champ 'messages' est requis")
    
    if not any(msg.role == "user" for msg in request.messages):
        raise HTTPException(status_code=400, detail="Au moins un message 'user' est requis")
    
    # Validation des tools si présents
    if request.tools:
        for tool in request.tools:
            if tool.type != "function":
                raise HTTPException(status_code=400, detail="Seul le type 'function' est supporté")
            
            func_def = tool.function
            if not func_def.name:
                raise HTTPException(status_code=400, detail="Le nom de la fonction est requis")


async def _process_chat_completion(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Traite une requête de chat completion"""
    try:
        # Attente du modèle
        if not await model_manager.wait_for_model(timeout=60):
            raise HTTPException(status_code=503, detail="Modèle non disponible")
        
        # Traitement par le modèle
        results = []
        async for result in model_manager.generate_response(request):
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            results.append(result)
        
        # Si streaming, on ne devrait pas arriver ici
        if request.stream:
            return results
        
        # Traitement des function calls si détectés
        final_result = results[-1] if results else {"content": "", "finish_reason": "stop"}
        
        if final_result.get("function_calls"):
            # Exécution des fonctions
            for func_call_data in final_result["function_calls"]:
                func_call = FunctionCall(
                    name=func_call_data["name"],
                    arguments=func_call_data.get("arguments", {})
                )
                
                # Exécution de la fonction
                func_result = await function_registry.execute_function(func_call)
                
                # Ajout du résultat aux messages pour un nouveau tour
                new_messages = request.messages + [
                    ChatMessage(
                        role="assistant",
                        tool_calls=[{
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": func_call.name,
                                "arguments": json.dumps(func_call.get_arguments_dict())
                            }
                        }]
                    ),
                    ChatMessage(
                        role="tool",
                        name=func_call.name,
                        content=json.dumps(func_result.result) if func_result.result else func_result.error,
                        tool_call_id=str(uuid.uuid4())
                    )
                ]
                
                # Nouvelle requête avec le résultat de la fonction
                new_request = request.model_copy()
                new_request.messages = new_messages
                new_request.tools = None  # Évite la récursion
                
                # Génération de la réponse finale
                final_results = []
                async for result in model_manager.generate_response(new_request):
                    final_results.append(result)
                
                final_result = final_results[-1] if final_results else final_result
        
        return final_result
        
    except Exception as e:
        logger.error(f"Erreur dans le traitement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _create_stream_generator(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Générateur pour le streaming des réponses"""
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_timestamp = int(time.time())
    
    try:
        # Attente du modèle
        if not await model_manager.wait_for_model(timeout=60):
            yield f"data: {json.dumps({'error': {'message': 'Modèle non disponible'}})}\n\n"
            return
        
        # Génération streaming
        chunk_index = 0
        async for result in model_manager.generate_response(request):
            if "error" in result:
                yield f"data: {json.dumps({'error': {'message': result['error']}})}\n\n"
                return
            
            # Format de chunk OpenAI
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created_timestamp,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(
                            content=result.get("delta", {}).get("content"),
                            role="assistant" if chunk_index == 0 else None
                        ),
                        finish_reason=result.get("finish_reason")
                    )
                ]
            )
            
            yield f"data: {chunk.model_dump_json()}\n\n"
            chunk_index += 1
        
        # Fin du stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Erreur dans le streaming: {e}")
        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Crée une completion de chat compatible avec l'API OpenAI
    
    Supporte:
    - Messages multi-modaux (texte + images)
    - Function calling
    - Streaming
    - Tous les paramètres OpenAI standard
    """
    # Validation
    await _validate_request(request)
    
    logger.info(f"Chat completion - Model: {request.model}, Messages: {len(request.messages)}, Stream: {request.stream}")
    
    try:
        if request.stream:
            # Mode streaming
            return EventSourceResponse(
                _create_stream_generator(request),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Pour Nginx
                }
            )
        else:
            # Mode non-streaming
            result = await _process_chat_completion(request)
            
            # Construction de la réponse OpenAI
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=result.get("content", ""),
                            tool_calls=result.get("function_calls")
                        ),
                        finish_reason=result.get("finish_reason", "stop")
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=0,  # TODO: Calculer les vrais tokens
                    completion_tokens=0,  # TODO: Calculer les vrais tokens
                    total_tokens=0  # TODO: Calculer les vrais tokens
                )
            )
            
            return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")


@router.get("/v1/chat/functions")
async def list_available_functions():
    """Liste les fonctions disponibles pour le function calling"""
    return {
        "functions": function_registry.get_function_definitions(),
        "total": len(function_registry.functions)
    }


@router.post("/v1/chat/functions/execute")
async def execute_function_directly(function_call: FunctionCall):
    """Exécute directement une fonction (pour debug/test)"""
    try:
        result = await function_registry.execute_function(function_call)
        return {
            "function": function_call.name,
            "arguments": function_call.get_arguments_dict(),
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))