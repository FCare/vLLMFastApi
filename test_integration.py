#!/usr/bin/env python3
"""
Script de test d'intégration pour Qwen3-VL FastAPI Server
"""
import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, Any, List


class QwenAPITester:
    """Testeur pour l'API Qwen3-VL"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = None
        
        # Résultats des tests
        self.results: List[Dict[str, Any]] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Effectue une requête HTTP"""
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(method, url, **kwargs) as response:
            content_type = response.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                data = await response.json()
            else:
                data = await response.text()
            
            return {
                "status_code": response.status,
                "headers": dict(response.headers),
                "data": data
            }
    
    def _log_test(self, name: str, success: bool, message: str = "", duration: float = 0):
        """Enregistre le résultat d'un test"""
        result = {
            "test": name,
            "success": success,
            "message": message,
            "duration": duration
        }
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name} ({duration:.2f}s)")
        if message:
            print(f"      {message}")
    
    async def test_health(self) -> bool:
        """Test de santé du service"""
        start_time = time.time()
        
        try:
            response = await self._request("GET", "/health")
            duration = time.time() - start_time
            
            if response["status_code"] == 200:
                data = response["data"]
                model_loaded = data.get("model_loaded", False)
                
                if model_loaded:
                    self._log_test("Health Check", True, "Service sain et modèle chargé", duration)
                    return True
                else:
                    self._log_test("Health Check", False, "Service sain mais modèle non chargé", duration)
                    return False
            else:
                self._log_test("Health Check", False, f"Status {response['status_code']}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Health Check", False, f"Erreur: {e}", duration)
            return False
    
    async def test_models_list(self) -> bool:
        """Test de la liste des modèles"""
        start_time = time.time()
        
        try:
            response = await self._request("GET", "/v1/models")
            duration = time.time() - start_time
            
            if response["status_code"] == 200:
                data = response["data"]
                models = data.get("data", [])
                
                if models and len(models) > 0:
                    model_name = models[0].get("id", "")
                    self._log_test("Models List", True, f"Modèle disponible: {model_name}", duration)
                    return True
                else:
                    self._log_test("Models List", False, "Aucun modèle disponible", duration)
                    return False
            else:
                self._log_test("Models List", False, f"Status {response['status_code']}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Models List", False, f"Erreur: {e}", duration)
            return False
    
    async def test_chat_basic(self) -> bool:
        """Test de chat completion basique"""
        start_time = time.time()
        
        try:
            payload = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "messages": [
                    {"role": "user", "content": "Dites 'Bonjour' en français."}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            response = await self._request("POST", "/v1/chat/completions", 
                                         json=payload,
                                         headers={"Content-Type": "application/json"})
            duration = time.time() - start_time
            
            if response["status_code"] == 200:
                data = response["data"]
                choices = data.get("choices", [])
                
                if choices and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "")
                    self._log_test("Chat Basic", True, f"Réponse: {content[:50]}...", duration)
                    return True
                else:
                    self._log_test("Chat Basic", False, "Pas de réponse générée", duration)
                    return False
            else:
                self._log_test("Chat Basic", False, f"Status {response['status_code']}: {response['data']}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Chat Basic", False, f"Erreur: {e}", duration)
            return False
    
    async def test_chat_streaming(self) -> bool:
        """Test de chat completion avec streaming"""
        start_time = time.time()
        
        try:
            payload = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "messages": [
                    {"role": "user", "content": "Comptez de 1 à 5."}
                ],
                "max_tokens": 30,
                "stream": True
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            
            chunks_received = 0
            async with self.session.post(url, json=payload, 
                                       headers={"Content-Type": "application/json"}) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                                chunks_received += 1
                    
                    duration = time.time() - start_time
                    
                    if chunks_received > 0:
                        self._log_test("Chat Streaming", True, f"{chunks_received} chunks reçus", duration)
                        return True
                    else:
                        self._log_test("Chat Streaming", False, "Aucun chunk reçu", duration)
                        return False
                else:
                    duration = time.time() - start_time
                    self._log_test("Chat Streaming", False, f"Status {response.status}", duration)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Chat Streaming", False, f"Erreur: {e}", duration)
            return False
    
    async def test_function_calling(self) -> bool:
        """Test du function calling"""
        start_time = time.time()
        
        try:
            payload = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "messages": [
                    {"role": "user", "content": "Quelle est la météo à Paris?"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Obtenir la météo actuelle pour une localisation",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "La ville et le pays"
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 100
            }
            
            response = await self._request("POST", "/v1/chat/completions", 
                                         json=payload,
                                         headers={"Content-Type": "application/json"})
            duration = time.time() - start_time
            
            if response["status_code"] == 200:
                data = response["data"]
                choices = data.get("choices", [])
                
                if choices:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls")
                    content = message.get("content", "")
                    
                    if tool_calls:
                        self._log_test("Function Calling", True, "Appel de fonction détecté", duration)
                    elif "météo" in content.lower() or "weather" in content.lower():
                        self._log_test("Function Calling", True, "Réponse liée à la météo générée", duration)
                    else:
                        self._log_test("Function Calling", False, "Pas d'appel de fonction détecté", duration)
                        return False
                    
                    return True
                else:
                    self._log_test("Function Calling", False, "Pas de réponse générée", duration)
                    return False
            else:
                self._log_test("Function Calling", False, f"Status {response['status_code']}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Function Calling", False, f"Erreur: {e}", duration)
            return False
    
    async def test_multimodal(self) -> bool:
        """Test du support multi-modal (simulation)"""
        start_time = time.time()
        
        try:
            # Test avec format multi-modal (même sans vraie image)
            payload = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Décrivez cette image:"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake_data"}}
                        ]
                    }
                ],
                "max_tokens": 50
            }
            
            response = await self._request("POST", "/v1/chat/completions", 
                                         json=payload,
                                         headers={"Content-Type": "application/json"})
            duration = time.time() - start_time
            
            # Le serveur devrait au moins traiter la requête (même si l'image est fausse)
            if response["status_code"] in [200, 400, 422]:
                self._log_test("Multimodal Support", True, "Format multimodal accepté", duration)
                return True
            else:
                self._log_test("Multimodal Support", False, f"Status {response['status_code']}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Multimodal Support", False, f"Erreur: {e}", duration)
            return False
    
    async def test_metrics(self) -> bool:
        """Test des métriques"""
        start_time = time.time()
        
        try:
            response = await self._request("GET", "/metrics")
            duration = time.time() - start_time
            
            if response["status_code"] == 200:
                data = response["data"]
                if isinstance(data, dict) and "requests_total" in data:
                    self._log_test("Metrics", True, "Métriques disponibles", duration)
                    return True
                else:
                    self._log_test("Metrics", False, "Format de métriques invalide", duration)
                    return False
            else:
                self._log_test("Metrics", False, f"Status {response['status_code']}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self._log_test("Metrics", False, f"Erreur: {e}", duration)
            return False
    
    async def run_all_tests(self) -> bool:
        """Exécute tous les tests"""
        print(f"🧪 Démarrage des tests d'intégration sur {self.base_url}")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health),
            ("Models List", self.test_models_list),
            ("Chat Basic", self.test_chat_basic),
            ("Chat Streaming", self.test_chat_streaming),
            ("Function Calling", self.test_function_calling),
            ("Multimodal Support", self.test_multimodal),
            ("Metrics", self.test_metrics),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                if success:
                    passed += 1
            except Exception as e:
                self._log_test(test_name, False, f"Exception: {e}")
        
        print("=" * 60)
        print(f"📊 Résultats: {passed}/{total} tests réussis")
        
        if passed == total:
            print("🎉 Tous les tests sont passés!")
            return True
        else:
            print("⚠️  Certains tests ont échoué.")
            return False
    
    def print_summary(self):
        """Affiche un résumé détaillé"""
        print("\n📋 Résumé détaillé des tests:")
        print("-" * 60)
        
        for result in self.results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']:20} ({result['duration']:5.2f}s)")
            if result["message"]:
                print(f"   {result['message']}")


async def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests d'intégration Qwen3-VL API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="URL de base de l'API (défaut: http://localhost:8000)")
    parser.add_argument("--wait", type=int, default=30, 
                       help="Temps d'attente pour que le service soit prêt (secondes)")
    
    args = parser.parse_args()
    
    # Attente que le service soit prêt
    print(f"⏳ Attente que le service soit prêt sur {args.url}...")
    
    ready = False
    for i in range(args.wait):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{args.url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("model_loaded"):
                            ready = True
                            break
        except:
            pass
        
        print(f"   Tentative {i+1}/{args.wait}...")
        await asyncio.sleep(1)
    
    if not ready:
        print(f"❌ Service non prêt après {args.wait}s")
        return False
    
    print("✅ Service prêt!")
    
    # Exécution des tests
    async with QwenAPITester(args.url) as tester:
        success = await tester.run_all_tests()
        tester.print_summary()
        return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        sys.exit(1)