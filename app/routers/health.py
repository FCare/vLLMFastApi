"""
Router pour les endpoints de santé et monitoring
"""
import time
import psutil
import torch
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..schemas.openai_schemas import HealthStatus, MetricsResponse
from ..models.qwen_model import model_manager
from ..utils.async_queue import queue_manager

logger = logging.getLogger(__name__)

# Création du router
router = APIRouter(tags=["health"])

# Métriques globales
_start_time = time.time()
_request_count = 0
_total_response_time = 0.0


def _get_gpu_info() -> Dict[str, Any]:
    """Récupère les informations GPU"""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "count": 0,
            "memory_used": 0.0,
            "memory_total": 0.0,
            "utilization": 0.0
        }
    
    try:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        
        # Mémoire GPU
        memory_used = torch.cuda.memory_allocated(current_device) / (1024**3)  # GB
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
        
        # Utilisation (approximation basée sur la mémoire)
        utilization = (memory_used / memory_total) * 100 if memory_total > 0 else 0
        
        return {
            "available": True,
            "count": gpu_count,
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
            "memory_used": round(memory_used, 2),
            "memory_total": round(memory_total, 2),
            "utilization": round(utilization, 1)
        }
    except Exception as e:
        logger.warning(f"Erreur lors de la récupération des infos GPU: {e}")
        return {
            "available": False,
            "error": str(e)
        }


def _get_system_memory() -> Dict[str, float]:
    """Récupère les informations mémoire système"""
    try:
        memory = psutil.virtual_memory()
        return {
            "used": round(memory.used / (1024**3), 2),  # GB
            "total": round(memory.total / (1024**3), 2),  # GB
            "percentage": round(memory.percent, 1)
        }
    except Exception as e:
        logger.warning(f"Erreur lors de la récupération de la mémoire: {e}")
        return {"used": 0.0, "total": 0.0, "percentage": 0.0}


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Vérification de santé du service
    
    Retourne l'état général du service, du modèle et des ressources
    """
    try:
        # Informations du modèle
        model_info = model_manager.get_model_info()
        model_loaded = model_info["ready"]
        
        # Informations GPU
        gpu_info = _get_gpu_info()
        gpu_available = gpu_info["available"]
        
        # Informations mémoire
        memory_info = _get_system_memory()
        
        # Temps de fonctionnement
        uptime = time.time() - _start_time
        
        # Statut général
        if model_loaded and gpu_available:
            status = "healthy"
        elif model_info["loading"]:
            status = "loading"
        else:
            status = "unhealthy"
        
        return HealthStatus(
            status=status,
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage={
                "system": memory_info,
                "gpu": {
                    "used": gpu_info.get("memory_used", 0),
                    "total": gpu_info.get("memory_total", 0),
                    "utilization": gpu_info.get("utilization", 0)
                }
            },
            uptime=uptime,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Erreur dans le health check: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la vérification de santé")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Métriques détaillées du service
    
    Retourne les statistiques de performance et d'utilisation
    """
    try:
        # Statistiques de la queue
        queue_stats = queue_manager.get_queue_stats()
        
        # Informations GPU
        gpu_info = _get_gpu_info()
        
        # Informations mémoire système
        memory_info = _get_system_memory()
        
        # Calcul des métriques de performance
        uptime = time.time() - _start_time
        requests_per_minute = (_request_count / (uptime / 60)) if uptime > 60 else 0
        average_response_time = (_total_response_time / _request_count) if _request_count > 0 else 0
        
        return MetricsResponse(
            requests_total=_request_count,
            requests_per_minute=round(requests_per_minute, 2),
            average_response_time=round(average_response_time, 3),
            active_connections=queue_stats.get("active_tasks", 0),
            queue_size=queue_stats.get("queue_size", 0),
            gpu_utilization=gpu_info.get("utilization", 0.0),
            memory_used=memory_info.get("used", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des métriques")


@router.get("/ready")
async def readiness_check():
    """
    Vérification de préparation (pour Kubernetes)
    
    Retourne 200 si le service est prêt à recevoir du trafic
    """
    try:
        model_info = model_manager.get_model_info()
        
        if model_info["ready"]:
            return {"status": "ready", "model": "loaded"}
        elif model_info["loading"]:
            raise HTTPException(status_code=503, detail="Modèle en cours de chargement")
        else:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur dans le readiness check: {e}")
        raise HTTPException(status_code=503, detail="Service non prêt")


@router.get("/live")
async def liveness_check():
    """
    Vérification de vie (pour Kubernetes)
    
    Retourne 200 si le service est vivant
    """
    try:
        # Vérification basique que l'application répond
        return {"status": "alive", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Erreur dans le liveness check: {e}")
        raise HTTPException(status_code=503, detail="Service non vivant")


@router.get("/status/detailed")
async def detailed_status():
    """
    Statut détaillé pour debugging
    
    Retourne toutes les informations disponibles
    """
    try:
        model_info = model_manager.get_model_info()
        queue_stats = queue_manager.get_queue_stats()
        gpu_info = _get_gpu_info()
        memory_info = _get_system_memory()
        
        # Informations du processus
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        return {
            "service": {
                "uptime": time.time() - _start_time,
                "requests_total": _request_count,
                "version": "1.0.0"
            },
            "model": model_info,
            "queue": queue_stats,
            "resources": {
                "gpu": gpu_info,
                "memory": memory_info,
                "cpu_percent": cpu_percent,
                "process_id": process.pid
            },
            "environment": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "torch_version": torch.__version__
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur dans le statut détaillé: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du statut")


# Middleware pour compter les requêtes et mesurer les temps de réponse
def track_request_metrics(start_time: float, status_code: int):
    """Met à jour les métriques de requête"""
    global _request_count, _total_response_time
    
    _request_count += 1
    response_time = time.time() - start_time
    _total_response_time += response_time