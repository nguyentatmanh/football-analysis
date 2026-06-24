import os
from typing import List, Dict
import numpy as np
from ultralytics import YOLO
from football_ai.detection.detector import Detector
from football_ai.detection.detection_result import DetectionResult
from football_ai.config.schema import DetectionConfig
from football_ai.detection.shared_inference import ModelRegistry, SharedInference

class FootballDetector(Detector):
    """
    Implementation of Detector using Ultralytics YOLOv8 weights.
    Supports running 3 separate models or 1 merged model.
    """
    
    # Roboflow Sports Default Class Mapping for the default model
    CLASS_MAP = {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee"
    }

    def __init__(self, config: DetectionConfig, device: str = "cpu"):
        """
        Initialize detector based on configured mode.
        """
        self.config = config
        self.device = device
        self.mode = getattr(config, "mode", "three_models")
        self.confidence = config.confidence
        self.iou = config.iou
        self.imgsz = config.imgsz

        print(f"[FootballDetector] Initializing in mode: {self.mode}")

        if self.mode == "merged_model":
            if not os.path.exists(config.merged_model_path):
                raise FileNotFoundError(
                    f"Merged model weights missing at: {config.merged_model_path}. "
                )
            # Load the merged model
            self.merged_model = ModelRegistry.get_model(config.merged_model_path, device)
        else:
            # Default three_models mode
            if not os.path.exists(config.player_model_path):
                raise FileNotFoundError(
                    f"Player model weights missing at: {config.player_model_path}. "
                    "Please run 'python scripts/download_assets.py' first."
                )
            self.player_model = ModelRegistry.get_model(config.player_model_path, device)
            
            # Ball model is optional, fall back if missing
            self.use_separate_ball_model = False
            if hasattr(config, "ball_model_path") and os.path.exists(config.ball_model_path):
                self.ball_model = ModelRegistry.get_model(config.ball_model_path, device)
                self.use_separate_ball_model = True
                print("[FootballDetector] Using dedicated ball detection model.")
            else:
                print("[FootballDetector] Dedicated ball detection model not found. Falling back to player model for ball detection.")

    def detect_frame(self, frame: np.ndarray, frame_index: int) -> List[DetectionResult]:
        """
        Detect soccer players, goalkeepers, referee, and balls in a video frame.
        """
        detections = []
        
        if self.mode == "merged_model":
            results = SharedInference.run_inference(
                self.config.merged_model_path,
                self.merged_model,
                frame,
                frame_index,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False
            )
            
            if len(results) == 0:
                return detections
                
            result = results[0]
            boxes = result.boxes
            names = getattr(self.merged_model, 'names', {})
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # Normalize role/class name dynamically from the merged model classes
                class_name_raw = names.get(cls_id, "unknown").lower()
                
                if "goalkeeper" in class_name_raw or class_name_raw == "gk":
                    role = "goalkeeper"
                elif "player" in class_name_raw:
                    role = "player"
                elif "referee" in class_name_raw:
                    role = "referee"
                elif "ball" in class_name_raw:
                    role = "ball"
                elif "pitch" in class_name_raw:
                    # Ignore pitch bounding boxes for player/object detection!
                    continue
                else:
                    # In merged mode, do not fallback to CLASS_MAP since it has different class IDs
                    continue
                
                if role in ["player", "goalkeeper", "referee", "ball"]:
                    det = DetectionResult(
                        frame_index=frame_index,
                        class_id=cls_id,
                        class_name=role,
                        confidence=conf,
                        bbox_xyxy=xyxy,
                        role=role
                    )
                    detections.append(det)
                    
        else:
            # 1. Run player model
            player_results = SharedInference.run_inference(
                self.config.player_model_path,
                self.player_model,
                frame,
                frame_index,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False
            )
            
            # Parse player model detections
            if len(player_results) > 0:
                result = player_results[0]
                boxes = result.boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    class_name = self.CLASS_MAP.get(cls_id, "unknown")
                    
                    # If using separate ball model, skip ball detections from the player model
                    if self.use_separate_ball_model and class_name == "ball":
                        continue
                        
                    if class_name in ["player", "goalkeeper", "referee", "ball"]:
                        det = DetectionResult(
                            frame_index=frame_index,
                            class_id=cls_id,
                            class_name=class_name,
                            confidence=conf,
                            bbox_xyxy=xyxy,
                            role=class_name
                        )
                        detections.append(det)
            
            # 2. Run ball model (if configured and exists)
            if self.use_separate_ball_model:
                ball_results = SharedInference.run_inference(
                    self.config.ball_model_path,
                    self.ball_model,
                    frame,
                    frame_index,
                    conf=self.confidence,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    verbose=False
                )
                
                if len(ball_results) > 0:
                    result = ball_results[0]
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].tolist()
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        
                        det = DetectionResult(
                            frame_index=frame_index,
                            class_id=cls_id,
                            class_name="ball",
                            confidence=conf,
                            bbox_xyxy=xyxy,
                            role="ball"
                        )
                        detections.append(det)
                        
        return detections
