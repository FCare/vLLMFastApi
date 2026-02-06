"""
Router pour les endpoints de modèles compatibles OpenAI
"""
import time
import logging
from fastapi import APIRouter, HTTPException

from ..schemas.openai_schemas import ModelsListResponse, ModelInfo
from ..models.qwen_model import model_manager

logger = logging.getLogger(__name__)

# Création du router
router = APIRouter(tags=["models"])


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """
    Liste les modèles disponibles - Compatible avec l'API OpenAI
    
    Retourne les informations sur le modèle Qwen3-VL chargé
    """
    try:
        model_info = model_manager.get_model_info()
        
        # Construction de la réponse compatible OpenAI
        models = [
            ModelInfo(
                id=model_info["model_name"],
                created=int(time.time() - (model_info.get("load_time", 0) or 0)),
                owned_by="Qwen",
                permission=[]
            )
        ]
        
        return ModelsListResponse(data=models)
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des modèles: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des modèles")


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    Récupère les détails d'un modèle spécifique
    """
    try:
        model_info = model_manager.get_model_info()
        
        if model_id != model_info["model_name"]:
            raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' non trouvé")
        
        return ModelInfo(
            id=model_info["model_name"],
            created=int(time.time() - (model_info.get("load_time", 0) or 0)),
            owned_by="Qwen",
            permission=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du modèle {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du modèle")


@router.get("/v1/models/status")
async def get_model_status():
    """
    Statut détaillé du modèle (extension non-OpenAI)
    """
    try:
        return model_manager.get_model_info()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du statut")