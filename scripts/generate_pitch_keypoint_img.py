import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Paths
model_path = "data/models/football-pitch-detection.pt"
video_path = "data/raw/sample_10s.mp4"
output_image_path = "pitch_keypoint_result.png"

print(f"Loading model: {model_path}")
model = YOLO(model_path)

print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

best_frame_idx = 0
best_frame = None
max_merged_keypoints = -1
best_merged_xy = None
best_merged_conf = None

print("Scanning video for the frame that yields the most merged high-confidence keypoints...")
pbar = tqdm(total=total_frames)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Check every 5th frame for speed
    if frame_idx % 5 == 0:
        results = model(frame, imgsz=1280, conf=0.3, verbose=False)
        if results and len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            result = results[0]
            
            # Merge keypoints from ALL detections in this frame
            temp_merged_xy = np.zeros((32, 2))
            temp_merged_conf = np.zeros(32)
            
            for i in range(len(result.keypoints)):
                xy = result.keypoints.xy[i].cpu().numpy()
                conf = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else np.ones(32)
                
                for j in range(32):
                    # If this detection has higher confidence for keypoint j, use it
                    if conf[j] > temp_merged_conf[j]:
                        temp_merged_conf[j] = conf[j]
                        temp_merged_xy[j] = xy[j]
            
            # Count how many merged keypoints have confidence > 0.5
            high_conf_count = np.sum((temp_merged_xy[:, 0] > 1.0) & (temp_merged_xy[:, 1] > 1.0) & (temp_merged_conf > 0.5))
            
            if high_conf_count > max_merged_keypoints:
                max_merged_keypoints = high_conf_count
                best_frame_idx = frame_idx
                best_frame = frame.copy()
                best_merged_xy = temp_merged_xy.copy()
                best_merged_conf = temp_merged_conf.copy()
                    
    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()

if best_frame is None:
    print("Error: Could not find any frames with pitch keypoints.")
    exit(1)

print(f"\nOptimal frame found at Frame {best_frame_idx} with {max_merged_keypoints} merged high-confidence keypoints.")

# Draw the merged keypoints on the best frame
annotated_frame = best_frame
h, w, _ = annotated_frame.shape

# Define modern BGR colors
color_glow = (0, 223, 252)       # Bright Golden Yellow
color_core = (0, 0, 255)         # Neon Red
color_text = (255, 255, 255)     # Pure White
color_bg = (40, 40, 40)          # Dark Charcoal

draw_count = 0
for idx, (x, y) in enumerate(best_merged_xy):
    confidence = best_merged_conf[idx]
    
    if x > 1.0 and y > 1.0 and confidence > 0.5:
        cx, cy = int(x), int(y)
        
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
print(f"Successfully generated merged {output_image_path} (from frame {best_frame_idx}) with {draw_count} keypoints!")
