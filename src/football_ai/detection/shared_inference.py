import numpy as np
from ultralytics import YOLO

class ModelRegistry:
    """
    Registry to load YOLO models once and reuse them across different detector classes.
    Prevents loading the same model weights multiple times in RAM/VRAM.
    """
    _models = {}

    @classmethod
    def get_model(cls, model_path: str, device: str = "cpu") -> YOLO:
        if model_path not in cls._models:
            print(f"[ModelRegistry] Loading YOLO model: {model_path} on {device}...")
            model = YOLO(model_path)
            model.to(device)
            cls._models[model_path] = model
        return cls._models[model_path]

    @classmethod
    def clear(cls):
        cls._models.clear()


class SharedInference:
    """
    Utility to cache YOLO inference results for the current frame index.
    If multiple components run the same model on the same frame, this avoids duplicate forward passes.
    """
    _cache = {}  # Key: model_path, Value: (frame_index, results)

    @classmethod
    def run_inference(cls, model_path: str, model: YOLO, frame: np.ndarray, frame_index: int, **kwargs):
        """
        Run inference using the specified model, caching the result based on model path and frame_index.
        """
        # If frame_index is negative, run without caching
        if frame_index < 0:
            return model(frame, **kwargs)

        if model_path in cls._cache:
            cached_frame_index, cached_results = cls._cache[model_path]
            if cached_frame_index == frame_index:
                # Cache hit!
                return cached_results

        # Cache miss, run forward pass
        results = model(frame, **kwargs)
        
        # Save to cache
        cls._cache[model_path] = (frame_index, results)
        return results

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
