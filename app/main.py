"""
Serveur FastAPI principal avec support Qwen3-VL, interface OpenAI et function calling
"""
import os
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from .routers import chat, models, health
from .routers.health import track_request_metrics
from .models.qwen_model import model_manager
from .utils.async_queue import queue_manager
from .schemas.openai_schemas import ErrorResponse, ErrorDetail

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration depuis l'environnement
API_PREFIX = os.getenv("API_PREFIX", "")
ROOT_PATH = os.getenv("ROOT_PATH", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))

# Configuration du modèle
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "32768"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

logger.info(f"Configuration - Préfixe API: '{API_PREFIX}', Root path: '{ROOT_PATH}'")
logger.info(f"Modèle: {MODEL_NAME}, Context: {MAX_SEQ_LENGTH}, 4-bit: {LOAD_IN_4BIT}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    logger.info("🚀 Démarrage du serveur Qwen3-VL FastAPI")
    
    # Configuration du gestionnaire de modèle
    model_manager.model_name = MODEL_NAME
    model_manager.max_seq_length = MAX_SEQ_LENGTH
    model_manager.load_in_4bit = LOAD_IN_4BIT
    
    # Démarrage du chargement du modèle en arrière-plan
    load_task = asyncio.create_task(model_manager.load_model())
    
    # Démarrage des workers de queue
    async def model_processor(request):
        """Processeur pour la queue asynchrone"""
        results = []
        async for result in model_manager.generate_response(request):
            results.append(result)
        return results[-1] if results else {"content": "", "finish_reason": "stop"}
    
    await queue_manager.start_workers(model_processor)
    
    logger.info("✅ Services démarrés - Chargement du modèle en cours...")
    
    yield
    
    # Nettoyage
    logger.info("🛑 Arrêt du serveur...")
    
    await queue_manager.stop_workers()
    await model_manager.cleanup()
    
    if not load_task.done():
        load_task.cancel()
        try:
            await load_task
        except asyncio.CancelledError:
            pass
    
    logger.info("✅ Nettoyage terminé")


# Création de l'application FastAPI
app = FastAPI(
    title="Qwen3-VL API Server",
    description="Serveur d'API compatible OpenAI utilisant Qwen3-VL avec Unsloth et support function calling",
    version="1.0.0",
    root_path=ROOT_PATH,
    lifespan=lifespan,
    docs_url=f"{API_PREFIX}/docs" if API_PREFIX else "/docs",
    redoc_url=f"{API_PREFIX}/redoc" if API_PREFIX else "/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json" if API_PREFIX else "/openapi.json"
)


# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Middleware de sécurité (optionnel)
if os.getenv("TRUSTED_HOSTS"):
    trusted_hosts = os.getenv("TRUSTED_HOSTS").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)


# Middleware de tracking des requêtes
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware pour tracker les requêtes et mesurer les performances"""
    start_time = time.time()
    
    # Traitement de la requête
    try:
        response = await call_next(request)
        track_request_metrics(start_time, response.status_code)
        
        # Headers de performance
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "1.0.0"
        
        return response
        
    except Exception as e:
        track_request_metrics(start_time, 500)
        logger.error(f"Erreur dans le middleware: {e}")
        raise


# Gestionnaires d'erreur globaux
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Gestionnaire pour les erreurs de validation"""
    logger.warning(f"Erreur de validation: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                message="Données de requête invalides",
                type="validation_error",
                code="invalid_request"
            )
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Gestionnaire pour les erreurs HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.detail,
                type="http_error",
                code=str(exc.status_code)
            )
        ).model_dump()
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Gestionnaire pour les erreurs serveur internes"""
    logger.error(f"Erreur serveur interne: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message="Erreur interne du serveur",
                type="internal_error",
                code="500"
            )
        ).model_dump()
    )


# Routes de base
@app.get("/")
async def root():
    """Page d'accueil avec redirection vers la documentation"""
    if API_PREFIX:
        return RedirectResponse(url=f"{API_PREFIX}/docs")
    return {
        "message": "Qwen3-VL API Server",
        "version": "1.0.0",
        "docs": f"{API_PREFIX}/docs" if API_PREFIX else "/docs",
        "openai_compatible": True,
        "features": [
            "Chat completions",
            "Function calling", 
            "Multi-modal (vision)",
            "Streaming",
            "Async processing"
        ]
    }


@app.get(f"{API_PREFIX}/")
async def api_root():
    """Page d'accueil pour le préfixe API"""
    return await root()


# Inclusion des routers avec préfixes
app.include_router(
    chat.router,
    prefix=API_PREFIX,
    tags=["chat"]
)

app.include_router(
    models.router,
    prefix=API_PREFIX,
    tags=["models"]
)

app.include_router(
    health.router,
    prefix=API_PREFIX,
    tags=["health"]
)


# Endpoints additionnels pour compatibility et debug
@app.get(f"{API_PREFIX}/v1/")
async def openai_api_root():
    """Point d'entrée de l'API v1 (compatibilité OpenAI)"""
    return {
        "message": "OpenAI Compatible API",
        "version": "v1",
        "endpoints": [
            f"{API_PREFIX}/v1/chat/completions",
            f"{API_PREFIX}/v1/models",
            f"{API_PREFIX}/health"
        ]
    }


@app.get(f"{API_PREFIX}/info")
async def server_info():
    """Informations sur le serveur"""
    return {
        "server": "Qwen3-VL FastAPI",
        "version": "1.0.0",
        "api_prefix": API_PREFIX,
        "root_path": ROOT_PATH,
        "model": {
            "name": MODEL_NAME,
            "max_context": MAX_SEQ_LENGTH,
            "quantized": LOAD_IN_4BIT
        },
        "features": {
            "openai_compatible": True,
            "function_calling": True,
            "multimodal": True,
            "streaming": True,
            "async_queue": True
        },
        "endpoints": {
            "chat": f"{API_PREFIX}/v1/chat/completions",
            "models": f"{API_PREFIX}/v1/models",
            "health": f"{API_PREFIX}/health",
            "docs": f"{API_PREFIX}/docs"
        }
    }


# Fonction principale pour lancement direct
def main():
    """Point d'entrée principal"""
    logger.info(f"Démarrage du serveur sur {HOST}:{PORT}")
    
    # Configuration Uvicorn
    uvicorn_config = {
        "app": "app.main:app",
        "host": HOST,
        "port": PORT,
        "workers": WORKERS,
        "log_level": "info",
        "access_log": True,
        "reload": os.getenv("RELOAD", "false").lower() == "true",
    }
    
    # Configuration SSL optionnelle
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    ssl_certfile = os.getenv("SSL_CERTFILE")
    if ssl_keyfile and ssl_certfile:
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        })
        logger.info("SSL configuré")
    
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()