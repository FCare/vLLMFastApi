"""
Gestionnaire de modèle Qwen3-VL avec vLLM pour performance optimale
"""
import asyncio
import os
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import json

try:
    from vllm import LLM, SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    logging.error("vLLM n'est pas disponible - installation requise: pip install vllm")
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer
from PIL import Image
import base64
import io

from ..schemas.openai_schemas import ChatMessage, ChatCompletionRequest

logger = logging.getLogger(__name__)


class QwenVLLMManager:
    """Gestionnaire du modèle Qwen3-VL avec vLLM"""
    
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
        self.max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
        self.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
        self.loading = False
        self.ready = False
        self.load_start_time = None
        
        # Semaphore pour limiter les inférences concurrentes 
        self.inference_semaphore = asyncio.Semaphore(1)
        
        logger.info(f"QwenVLLMManager initialisé avec {self.model_name}")
        
    async def load_model(self) -> None:
        """Charge le modèle vLLM de manière asynchrone"""
        if self.loading or self.ready:
            return
            
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM n'est pas disponible - installation requise")
        
        self.loading = True
        self.load_start_time = time.time()
        logger.info(f"🚀 Chargement du modèle {self.model_name} avec vLLM")
        
        try:
            # Chargement dans un thread pool pour éviter le blocage
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            
            self.ready = True
            load_time = time.time() - self.load_start_time
            logger.info(f"✅ Modèle vLLM chargé avec succès en {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement vLLM: {e}")
            raise
        finally:
            self.loading = False
            
    def _load_model_sync(self) -> None:
        """Chargement synchrone du modèle vLLM"""
        try:
            # Configuration vLLM conservative pour éviter les segfaults
            vllm_args = {
                "model": self.model_name,
                "max_model_len": min(self.max_model_len, 4096),  # Limite plus conservatrice
                "gpu_memory_utilization": min(self.gpu_memory_utilization, 0.7),  # Mémoire réduite
                "trust_remote_code": True,  # Requis pour Qwen2-VL
                "dtype": "half",  # float16 pour économiser la mémoire
                "enforce_eager": True,  # Mode eager pour éviter les optimisations problématiques
                "disable_custom_all_reduce": True,  # Désactiver les optimisations
                "swap_space": 4,  # Espace de swap pour la mémoire
            }
            
            logger.info(f"Configuration vLLM: {vllm_args}")
            
            # Initialisation du modèle vLLM
            self.llm = LLM(**vllm_args)
            
            # Chargement du tokenizer pour les utilitaires
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info(f"✅ vLLM initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur dans _load_model_sync: {e}")
            raise
    
    async def wait_for_model(self, timeout: float = 300.0) -> bool:
        """Attend que le modèle soit prêt"""
        start_time = time.time()
        while not self.ready and (time.time() - start_time) < timeout:
            if not self.loading:
                asyncio.create_task(self.load_model())
            await asyncio.sleep(1)
        return self.ready
    
    def _prepare_messages_for_qwen(
        self, 
        messages: List[ChatMessage], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Prépare les messages au format Qwen3-VL"""
        conversation_parts = []
        
        # Message système avec outils si fournis
        system_content = "Tu es Qwen, un assistant IA utile créé par Alibaba Cloud."
        if tools:
            tools_description = "\n\nOutils disponibles:\n"
            for tool in tools:
                func = tool["function"]
                tools_description += f"- {func['name']}: {func.get('description', '')}\n"
            system_content += tools_description
            system_content += "\nPour utiliser un outil, réponds avec un appel de fonction au format JSON."
        
        conversation_parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")
        
        # Conversion des messages
        for msg in messages:
            role = msg.role
            content = ""
            
            if isinstance(msg.content, list):
                # Support multimodal : traitement des images et texte
                for part in msg.content:
                    if part.get("type") == "text":
                        content += part["text"]
                    elif part.get("type") == "image_url":
                        # Pour vLLM avec Qwen3-VL, nous incluons l'image
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:"):
                            content += f"\n<image>{image_url}</image>\n"
                        else:
                            content += f"\n<image>{image_url}</image>\n"
            else:
                content = msg.content or ""
            
            conversation_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Ajout du token assistant pour la génération
        conversation_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(conversation_parts)
    
    async def generate_response(
        self, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Génère une réponse avec vLLM de manière asynchrone"""
        if not self.ready:
            await self.wait_for_model()
        
        async with self.inference_semaphore:
            try:
                # Préparation du prompt
                prompt = self._prepare_messages_for_qwen(
                    request.messages, 
                    request.tools
                )
                
                # Configuration des paramètres de génération
                sampling_params = SamplingParams(
                    temperature=request.temperature or 0.7,
                    max_tokens=request.max_tokens or 1024,
                    top_p=request.top_p or 0.8,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    skip_special_tokens=False
                )
                
                # Génération avec vLLM
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None, 
                    self._generate_sync, 
                    prompt, 
                    sampling_params,
                    request.stream
                )
                
                if request.stream:
                    # Mode streaming : émulation pour l'instant
                    # vLLM supporte le streaming mais nécessite AsyncLLMEngine
                    generated_text = outputs[0].outputs[0].text
                    words = generated_text.split()
                    
                    for i, word in enumerate(words):
                        chunk = {
                            "delta": {"content": word + " " if i < len(words) - 1 else word},
                            "finish_reason": "stop" if i == len(words) - 1 else None
                        }
                        yield chunk
                        await asyncio.sleep(0.01)
                else:
                    # Mode non-streaming
                    generated_text = outputs[0].outputs[0].text.strip()
                    
                    # Analyse pour détecter les function calls
                    function_calls = self._extract_function_calls(generated_text)
                    
                    yield {
                        "content": generated_text,
                        "function_calls": function_calls,
                        "finish_reason": "stop"
                    }
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors de la génération vLLM: {e}")
                yield {
                    "error": str(e),
                    "type": "generation_error"
                }
    
    def _generate_sync(
        self, 
        prompt: str, 
        sampling_params: SamplingParams,
        stream: bool = False
    ):
        """Génération synchrone avec vLLM"""
        try:
            # Génération avec vLLM
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs
            
        except Exception as e:
            logger.error(f"❌ Erreur dans _generate_sync: {e}")
            raise
    
    def _extract_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les appels de fonction du texte généré"""
        function_calls = []
        
        # Pattern pour détecter les JSON de function calls
        import re
        json_pattern = r'\{[^}]*"name"\s*:\s*"[^"]+"[^}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                func_call = json.loads(match)
                if "name" in func_call:
                    function_calls.append(func_call)
            except json.JSONDecodeError:
                continue
        
        return function_calls
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        return {
            "model_name": self.model_name,
            "ready": self.ready,
            "loading": self.loading,
            "backend": "vLLM",
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "multimodal": True,
            "supports_function_calling": True,
            "load_time": time.time() - self.load_start_time if self.load_start_time else None
        }
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        logger.info("Ressources vLLM nettoyées")


# Instance globale du gestionnaire de modèle
model_manager = QwenVLLMManager()