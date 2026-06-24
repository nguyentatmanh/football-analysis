import cv2
import numpy as np
from ultralytics import YOLO

# Paths
model_path = "data/models/football-pitch-detection.pt"
video_path = "data/raw/sample_10s.mp4"
output_image_path = "test.png"

print(f"Loading new pitch detection model: {model_path}")
model = YOLO(model_path)

print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

# Skip to Frame 140
frame_to_use = 140
for _ in range(frame_to_use):
    ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video frame.")
    exit(1)

print(f"Running keypoint detection on Frame {frame_to_use} at imgsz=640...")
results = model(frame, imgsz=640, conf=0.3, verbose=False)

if not results or len(results) == 0 or not hasattr(results[0], 'keypoints') or results[0].keypoints is None or len(results[0].keypoints) == 0:
    print("Error: Could not detect keypoints.")
    exit(1)

result = results[0]

# Select the detection index corresponding to the largest bounding box area (full pitch)
pitch_idx = -1
if result.boxes is not None and len(result.boxes) > 0:
    max_area = -1
    for i, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        if area > max_area:
            max_area = area
            pitch_idx = i

if pitch_idx == -1:
    print("No pitch detection bounding boxes found.")
    exit(1)

print(f"Selected pitch detection index {pitch_idx} (largest bounding box).")

best_keypoints_xy = result.keypoints.xy[pitch_idx].cpu().numpy()
best_keypoints_conf = result.keypoints.conf[pitch_idx].cpu().numpy() if result.keypoints.conf is not None else np.ones(32)

annotated_frame = frame.copy()
h, w, _ = annotated_frame.shape

# Define BGR colors
color_glow = (0, 223, 252)       # Bright Golden Yellow
color_core = (0, 0, 255)         # Neon Red
color_text = (255, 255, 255)     # Pure White
color_bg = (40, 40, 40)          # Dark Charcoal

draw_count = 0
print("\n--- Detected Keypoints ---")
for idx, (x, y) in enumerate(best_keypoints_xy):
    confidence = best_keypoints_conf[idx]
    
    # Check if keypoint is detected and has high confidence (>0.5)
    if x > 1.0 and y > 1.0 and confidence > 0.5:
        cx, cy = int(x), int(y)
        print(f"Keypoint {idx}: ({cx}, {cy}) | Conf: {confidence:.3f}")
        
        # Outer glow ring
        cv2.circle(annotated_frame, (cx, cy), 12, color_glow, 2, cv2.LINE_AA)
        # Core center circle
        cv2.circle(annotated_frame, (cx, cy), 5, color_core, -1, cv2.LINE_AA)
        
        # Label ID + Confidence
        text = f"KP {idx} ({int(confidence * 100)}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        tx = cx + 15
        ty = cy - 5
        if tx + tw > w:
            tx = cx - tw - 15
            
        cv2.rectangle(
            annotated_frame,
            (tx - 3, ty - th - 3),
            (tx + tw + 3, ty + baseline + 3),
            color_bg,
            -1
        )
        cv2.rectangle(
            annotated_frame,
            (tx - 3, ty - th - 3),
            (tx + tw + 3, ty + baseline + 3),
            color_glow,
            1
        )
        cv2.putText(
            annotated_frame,
            text,
            (tx, ty),
            font,
            font_scale,
            color_text,
            thickness,
            cv2.LINE_AA
        )
        draw_count += 1

cv2.imwrite(output_image_path, annotated_frame)
print(f"\nSuccessfully generated {output_image_path} with {draw_count} high-confidence keypoints!")
