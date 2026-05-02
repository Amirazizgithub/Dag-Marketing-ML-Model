import cachetools
from models.dag_engine import CausalDAGEngine


class ModelManager:
    def __init__(self, max_clients: int = 50):
        self.cache = cachetools.LRUCache(maxsize=max_clients)

    def get_engine(self, client_number: int) -> CausalDAGEngine:
        """Retrieves or loads the engine."""
        if client_number not in self.cache:
            return self.warm_cache(client_number)
        return self.cache[client_number]

    def warm_cache(self, client_number: int) -> CausalDAGEngine:
        """
        Force-loads the engine into memory.
        Used during training to ensure the next API call is instant.
        """
        print(f"--- Warming Cache for client {client_number} ---")
        engine = CausalDAGEngine(data={"client_number": client_number})
        engine.load()  # The slow GCS call
        self.cache[client_number] = engine
        return engine


# Global instance
model_manager = ModelManager()
