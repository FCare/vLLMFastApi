"""
Schémas pour le function calling avec Qwen
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json


class FunctionCall(BaseModel):
    """Appel de fonction standard"""
    name: str = Field(description="Nom de la fonction")
    arguments: Union[str, Dict[str, Any]] = Field(description="Arguments de la fonction")
    
    def get_arguments_dict(self) -> Dict[str, Any]:
        """Retourne les arguments sous forme de dictionnaire"""
        if isinstance(self.arguments, str):
            try:
                return json.loads(self.arguments)
            except json.JSONDecodeError:
                return {"raw": self.arguments}
        return self.arguments


class FunctionResult(BaseModel):
    """Résultat d'exécution d'une fonction"""
    name: str = Field(description="Nom de la fonction exécutée")
    result: Any = Field(description="Résultat de l'exécution")
    error: Optional[str] = Field(None, description="Erreur si échec")
    execution_time: Optional[float] = Field(None, description="Temps d'exécution en secondes")


class QwenFunctionCall(BaseModel):
    """Format spécifique Qwen pour function calling"""
    name: str
    arguments: str  # JSON stringifié selon doc Qwen
    
    @classmethod
    def from_function_call(cls, func_call: FunctionCall) -> "QwenFunctionCall":
        """Convertit un FunctionCall vers le format Qwen"""
        if isinstance(func_call.arguments, dict):
            arguments_str = json.dumps(func_call.arguments, ensure_ascii=False)
        else:
            arguments_str = str(func_call.arguments)
        
        return cls(name=func_call.name, arguments=arguments_str)


class QwenToolMessage(BaseModel):
    """Message d'outil au format Qwen"""
    role: str = "tool"
    name: str
    content: str
    
    @classmethod
    def from_function_result(cls, result: FunctionResult) -> "QwenToolMessage":
        """Crée un message d'outil depuis un résultat de fonction"""
        if result.error:
            content = f"Erreur lors de l'exécution de {result.name}: {result.error}"
        else:
            content = json.dumps(result.result, ensure_ascii=False) if result.result else ""
        
        return cls(name=result.name, content=content)


# Fonctions d'exemple pour démonstration
EXAMPLE_FUNCTIONS = [
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
                        "description": "La ville et le pays, ex: Paris, France"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unité de température"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_web",
            "description": "Rechercher des informations sur le web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Nombre de résultats à retourner",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image", 
            "description": "Analyser le contenu d'une image",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "URL de l'image à analyser"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["objects", "text", "faces", "general"],
                        "description": "Type d'analyse à effectuer"
                    }
                },
                "required": ["image_url"]
            }
        }
    }
]


class FunctionRegistry:
    """Registre des fonctions disponibles"""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.executors: Dict[str, callable] = {}
    
    def register_function(self, func_def: Dict[str, Any], executor: callable):
        """Enregistre une fonction et son exécuteur"""
        func_name = func_def["function"]["name"]
        self.functions[func_name] = func_def
        self.executors[func_name] = executor
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Retourne toutes les définitions de fonctions"""
        return list(self.functions.values())
    
    async def execute_function(self, func_call: FunctionCall) -> FunctionResult:
        """Exécute une fonction"""
        import time
        start_time = time.time()
        
        if func_call.name not in self.executors:
            return FunctionResult(
                name=func_call.name,
                result=None,
                error=f"Fonction '{func_call.name}' non trouvée"
            )
        
        try:
            executor = self.executors[func_call.name]
            args = func_call.get_arguments_dict()
            
            # Exécution asynchrone si possible
            if hasattr(executor, '__call__') and hasattr(executor, '__await__'):
                result = await executor(**args)
            else:
                result = executor(**args)
            
            execution_time = time.time() - start_time
            
            return FunctionResult(
                name=func_call.name,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return FunctionResult(
                name=func_call.name,
                result=None,
                error=str(e),
                execution_time=execution_time
            )