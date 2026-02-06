"""
Gestionnaire de queue asynchrone pour les requêtes d'inférence
"""
import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """États possibles d'une tâche"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Tâche dans la queue"""
    id: str
    request: Any
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def processing_time(self) -> Optional[float]:
        """Temps de traitement en secondes"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def total_time(self) -> Optional[float]:
        """Temps total depuis la création"""
        if self.completed_at:
            return self.completed_at - self.created_at
        return time.time() - self.created_at


class AsyncQueueManager:
    """Gestionnaire de queue asynchrone pour l'inférence"""
    
    def __init__(self, max_queue_size: int = 100, max_workers: int = 1):
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        
        # Queue des tâches en attente
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Stockage des tâches par ID
        self.tasks: Dict[str, Task] = {}
        
        # Events pour notifier les résultats
        self.task_events: Dict[str, asyncio.Event] = {}
        
        # Workers
        self.workers: list = []
        self.running = False
        
        # Statistiques
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "queue_wait_time": 0.0
        }
        
        # Nettoyage automatique des anciennes tâches
        self.cleanup_interval = 300  # 5 minutes
        self.max_task_age = 3600  # 1 heure
        
        logger.info(f"AsyncQueueManager initialisé - Queue: {max_queue_size}, Workers: {max_workers}")
    
    async def start_workers(self, processor_func: Callable[[Any], Awaitable[Any]]):
        """Démarre les workers de traitement"""
        if self.running:
            return
        
        self.running = True
        self.processor_func = processor_func
        
        # Création des workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Démarrage du nettoyage automatique
        cleanup_task = asyncio.create_task(self._cleanup_old_tasks())
        self.workers.append(cleanup_task)
        
        logger.info(f"{self.max_workers} workers démarrés")
    
    async def stop_workers(self):
        """Arrête les workers"""
        if not self.running:
            return
        
        self.running = False
        
        # Annulation de tous les workers
        for worker in self.workers:
            worker.cancel()
        
        # Attente de l'arrêt
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Workers arrêtés")
    
    async def submit_task(self, request: Any, priority: int = 0) -> str:
        """Soumet une tâche à la queue"""
        task_id = str(uuid.uuid4())
        
        # Vérification de la capacité de la queue
        if self.task_queue.full():
            raise asyncio.QueueFull("Queue pleine, réessayez plus tard")
        
        # Création de la tâche
        task = Task(
            id=task_id,
            request=request,
            status=TaskStatus.QUEUED,
            created_at=time.time()
        )
        
        # Stockage
        self.tasks[task_id] = task
        self.task_events[task_id] = asyncio.Event()
        
        # Ajout à la queue
        await self.task_queue.put((priority, task_id, request))
        
        self.stats["total_requests"] += 1
        logger.debug(f"Tâche {task_id} ajoutée à la queue")
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Récupère le résultat d'une tâche"""
        if task_id not in self.tasks:
            raise ValueError(f"Tâche {task_id} introuvable")
        
        task = self.tasks[task_id]
        
        # Si déjà terminée, retourne immédiatement
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return self._format_task_result(task)
        
        # Attente du résultat
        try:
            event = self.task_events[task_id]
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._format_task_result(self.tasks[task_id])
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout en attendant le résultat de {task_id}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'une tâche"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "id": task.id,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "processing_time": task.processing_time,
            "total_time": task.total_time
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Annule une tâche"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.QUEUED:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self.task_events[task_id].set()
            logger.debug(f"Tâche {task_id} annulée")
            return True
        
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la queue"""
        active_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PROCESSING]
        queued_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.QUEUED]
        
        return {
            "queue_size": len(queued_tasks),
            "active_tasks": len(active_tasks),
            "total_tasks": len(self.tasks),
            "max_queue_size": self.max_queue_size,
            "workers": self.max_workers,
            "running": self.running,
            **self.stats
        }
    
    async def _worker(self, worker_name: str):
        """Worker de traitement des tâches"""
        logger.info(f"Worker {worker_name} démarré")
        
        while self.running:
            try:
                # Récupération d'une tâche
                priority, task_id, request = await self.task_queue.get()
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                # Vérification si la tâche est toujours valide
                if task.status != TaskStatus.QUEUED:
                    self.task_queue.task_done()
                    continue
                
                # Démarrage du traitement
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                
                logger.debug(f"Worker {worker_name} traite la tâche {task_id}")
                
                try:
                    # Traitement de la requête
                    result = await self.processor_func(request)
                    
                    # Succès
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()
                    
                    self.stats["completed_requests"] += 1
                    self._update_average_processing_time(task.processing_time)
                    
                    logger.debug(f"Tâche {task_id} completée en {task.processing_time:.2f}s")
                    
                except Exception as e:
                    # Erreur
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
                    
                    self.stats["failed_requests"] += 1
                    
                    logger.error(f"Erreur dans la tâche {task_id}: {e}")
                
                finally:
                    # Notification du résultat
                    if task_id in self.task_events:
                        self.task_events[task_id].set()
                    
                    self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans le worker {worker_name}: {e}")
        
        logger.info(f"Worker {worker_name} arrêté")
    
    async def _cleanup_old_tasks(self):
        """Nettoyage périodique des anciennes tâches"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                expired_tasks = []
                
                for task_id, task in self.tasks.items():
                    if (current_time - task.created_at) > self.max_task_age:
                        expired_tasks.append(task_id)
                
                # Suppression des tâches expirées
                for task_id in expired_tasks:
                    del self.tasks[task_id]
                    if task_id in self.task_events:
                        del self.task_events[task_id]
                
                if expired_tasks:
                    logger.info(f"Nettoyage de {len(expired_tasks)} tâches expirées")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans le nettoyage: {e}")
    
    def _format_task_result(self, task: Task) -> Dict[str, Any]:
        """Formate le résultat d'une tâche"""
        return {
            "id": task.id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "processing_time": task.processing_time,
            "total_time": task.total_time,
            "created_at": task.created_at,
            "completed_at": task.completed_at
        }
    
    def _update_average_processing_time(self, processing_time: Optional[float]):
        """Met à jour le temps de traitement moyen"""
        if processing_time is None:
            return
        
        completed = self.stats["completed_requests"]
        if completed == 1:
            self.stats["average_processing_time"] = processing_time
        else:
            # Moyenne mobile
            current_avg = self.stats["average_processing_time"]
            self.stats["average_processing_time"] = (
                (current_avg * (completed - 1) + processing_time) / completed
            )


# Instance globale du gestionnaire de queue
queue_manager = AsyncQueueManager()