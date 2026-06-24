import time
import json
import os

class PerformanceTracker:
    """
    Performance utility to track pipeline and inference runtime, FPS, 
    and save metrics for speed comparison.
    """
    def __init__(self, mode: str, device: str):
        self.mode = mode
        self.device = device
        self.start_time = 0.0
        self.end_time = 0.0
        self.total_frames = 0
        self.inference_time = 0.0

    def start(self):
        """Start tracking the session."""
        self.start_time = time.time()
        self.inference_time = 0.0

    def record_inference(self, duration: float):
        """Accumulate core model inference duration."""
        self.inference_time += duration

    def stop(self, total_frames: int):
        """End tracking session."""
        self.end_time = time.time()
        self.total_frames = total_frames

    def get_metrics(self) -> dict:
        """Compute average FPS and speedup factors."""
        total_time = self.end_time - self.start_time
        if total_time <= 0:
            total_time = 0.001
            
        avg_fps = self.total_frames / total_time
        
        inf_time = self.inference_time
        if inf_time <= 0:
            inf_time = 0.001
            
        inf_fps = self.total_frames / inf_time
        
        return {
            "model_mode": self.mode,
            "device": self.device,
            "total_frames": self.total_frames,
            "total_time_seconds": round(total_time, 2),
            "inference_time_seconds": round(self.inference_time, 2),
            "average_fps": round(avg_fps, 2),
            "inference_fps": round(inf_fps, 2)
        }

    def save(self, output_dir: str):
        """Save computed performance statistics to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        metrics = self.get_metrics()
        output_path = os.path.join(output_dir, "perf_metrics.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[PerformanceTracker] Saved performance metrics to: {output_path}")
