import os
import numpy as np
import supervision as sv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from football_ai.config.schema import DetectionConfig
from football_ai.detection.shared_inference import ModelRegistry, SharedInference

@dataclass
class PitchKeypointResult:
    frame_index: int
    keypoints_xy: Optional[List[List[float]]] = None # Shape: (32, 2)
    confidence: Optional[List[float]] = None        # Shape: (32,)
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PitchKeypointDetector:
    """
    Handles loading the specialized YOLO keypoint model and extracting
    predefined keypoint coordinates from video frames.
    Supports running a dedicated pitch keypoints model or a merged model.
    """
    def __init__(self, config: DetectionConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.mode = getattr(config, "mode", "three_models")
        self.conf_threshold = config.confidence

        if self.mode == "merged_model":
            is_pose_model = False
            if os.path.exists(config.merged_model_path):
                try:
                    temp_model = ModelRegistry.get_model(config.merged_model_path, device)
                    if getattr(temp_model, 'task', 'detect') == 'pose':
                        is_pose_model = True
                except Exception:
                    pass
            
            if is_pose_model:
                self.model_path = config.merged_model_path
                self.model = ModelRegistry.get_model(config.merged_model_path, device)
                print("[PitchKeypointDetector] Merged model supports pose task. Using it for keypoints.")
            else:
                if not os.path.exists(config.pitch_model_path):
                    raise FileNotFoundError(
                        f"Pitch keypoint model weights missing at: {config.pitch_model_path}."
                    )
                self.model_path = config.pitch_model_path
                self.model = ModelRegistry.get_model(config.pitch_model_path, device)
                print("[PitchKeypointDetector] Merged model is a detection model (no pose). Falling back to dedicated pitch keypoints model.")
        else:
            if not os.path.exists(config.pitch_model_path):
                raise FileNotFoundError(
                    f"Pitch keypoint model weights missing at: {config.pitch_model_path}. "
                    "Please ensure assets are downloaded."
                )
            self.model_path = config.pitch_model_path
            self.model = ModelRegistry.get_model(config.pitch_model_path, device)

    def detect(self, frame: np.ndarray, frame_index: int) -> PitchKeypointResult:
        """
        Execute pose detection and parse the pitch boundary landmarks.
        """
        results = SharedInference.run_inference(
            self.model_path,
            self.model,
            frame,
            frame_index,
            conf=self.conf_threshold,
            verbose=False
        )
        
        if not results:
            return PitchKeypointResult(frame_index=frame_index, success=False)
            
        result = results[0]
        # In YOLO pose, if keypoints are found
        if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints) == 0:
            return PitchKeypointResult(frame_index=frame_index, success=False)
            
        # We need to find the correct index corresponding to the pitch keypoints
        pitch_idx = -1
        
        if self.mode == "merged_model":
            # 1. Identify pitch detection by class name if possible
            names = getattr(self.model, 'names', {})
            pitch_cls_indices = [idx for idx, name in names.items() if 'pitch' in name.lower() or 'keypoint' in name.lower()]
            
            if pitch_cls_indices and hasattr(result, 'boxes') and result.boxes is not None:
                for i, cls_id in enumerate(result.boxes.cls.tolist()):
                    if int(cls_id) in pitch_cls_indices:
                        pitch_idx = i
                        break
            
            # 2. Fallback: Find the detection that has the most non-zero keypoints (more than 4 valid coordinates)
            if pitch_idx == -1:
                xy_np = result.keypoints.xy.cpu().numpy() # Shape: (num_detections, num_points, 2)
                max_detected = -1
                for i in range(len(xy_np)):
                    valid_count = np.sum((xy_np[i, :, 0] > 1.0) & (xy_np[i, :, 1] > 1.0))
                    if valid_count > max_detected:
                        max_detected = valid_count
                        pitch_idx = i
                        
                # If the max detected keypoints is less than 4, it's not a valid pitch
                if max_detected < 4:
                    pitch_idx = -1
        else:
            # Dedicated pitch model only has 1 detection of pitch class, index 0.
            pitch_idx = 0

        if pitch_idx == -1 or pitch_idx >= len(result.keypoints):
            return PitchKeypointResult(frame_index=frame_index, success=False)

        xy = result.keypoints.xy[pitch_idx].cpu().numpy() # (32, 2)
        conf = result.keypoints.conf[pitch_idx].cpu().numpy() if result.keypoints.conf is not None else None

        # Validation check: do we have any non-zero detected points?
        # Valid keypoints are non-zero.
        valid_mask = (xy[:, 0] > 1.0) & (xy[:, 1] > 1.0)
        num_detected = np.sum(valid_mask)

        # A homography estimation requires at least 4 points
        if num_detected < 4:
            return PitchKeypointResult(
                frame_index=frame_index,
                keypoints_xy=xy.tolist(),
                confidence=conf.tolist() if conf is not None else None,
                success=False
            )

        return PitchKeypointResult(
            frame_index=frame_index,
            keypoints_xy=xy.tolist(),
            confidence=conf.tolist() if conf is not None else None,
            success=True
        )
