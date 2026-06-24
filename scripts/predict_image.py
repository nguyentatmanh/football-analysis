import os
import sys
import argparse
import cv2
from ultralytics import YOLO

def draw_beautiful_detections(img, boxes):
    # Color palette (BGR)
    colors = {
        0: (0, 165, 255),    # Ball: Orange
        1: (255, 0, 180),    # Goalkeeper: Magenta/Pink
        2: (0, 220, 0),      # Player: Vibrant Green
        3: (0, 0, 255),      # Referee: Red
        4: (255, 255, 0)     # Pitch: Solid Neon Cyan (Electric Cyan)
    }
    
    names = {
        0: "Ball",
        1: "Goalkeeper",
        2: "Player",
        3: "Referee",
        4: "Pitch"
    }

    # Bolder drawing settings
    thickness_pitch = 6
    thickness_object = 3
    font_scale = 0.75
    font_thickness = 2

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        
        color = colors.get(cls_id, (255, 255, 255))
        class_name = names.get(cls_id, "Unknown")
        
        if cls_id == 4: # Pitch
            # Bolder pitch rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness_pitch)
            # Label
            label = f"{class_name} ({conf:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(img, (x1, y1 - h - 12), (x1 + w + 16, y1), color, -1)
            cv2.putText(img, label, (x1 + 8, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        else:
            # Draw standard objects (Player, Referee, Goalkeeper, Ball)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness_object)
            
            # Larger, clear text label
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.15, font_thickness)
            
            # Draw tag background
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 12, y1), color, -1)
            text_color = (0, 0, 0) if cls_id in [2, 4] else (255, 255, 255)
            cv2.putText(img, label, (x1 + 6, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.15, text_color, font_thickness, cv2.LINE_AA)

    return img

def main():
    parser = argparse.ArgumentParser(description="Predict and save annotated image using best.pt")
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to an image file. If not provided, will automatically scan for the best frame"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="prediction_result.png",
        help="Path to save the annotated output image"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.3,
        help="Confidence threshold"
    )
    args = parser.parse_args()

    model_path = "data/models/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Use GPU if available
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    print(f"Loading model: {model_path} onto {device}")
    model = YOLO(model_path)

    # Resolve image source
    frame = None
    results = None
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            sys.exit(1)
        print(f"Reading image: {args.image}")
        frame = cv2.imread(args.image)
        # Run inference once
        results = model(frame, conf=args.conf, imgsz=1280, device=device)
    else:
        video_path = "data/raw/sample_10s.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            sys.exit(1)
            
        print(f"Scanning video for the best pitch detection frame: {video_path}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        best_frame_idx = 0
        max_pitch_conf = -1.0
        best_results = None
        
        # Scan every 10th frame to be fast
        step = 10
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret or f is None:
                break
                
            # Quick inference
            res = model(f, conf=args.conf, imgsz=1280, verbose=False, device=device)
            if len(res) > 0 and len(res[0].boxes) > 0:
                # Find if there is a pitch box
                pitch_boxes = [box for box in res[0].boxes if int(box.cls[0].item()) == 4]
                if pitch_boxes:
                    # Get the maximum confidence for pitch in this frame
                    conf_val = max(float(box.conf[0].item()) for box in pitch_boxes)
                    if conf_val > max_pitch_conf:
                        max_pitch_conf = conf_val
                        best_frame_idx = idx
                        best_results = res
        
        if max_pitch_conf > 0:
            print(f"Best pitch detection found at Frame {best_frame_idx} with Confidence: {max_pitch_conf:.2f}")
        else:
            print("No pitch detection found during scan, falling back to Frame 0")
            best_frame_idx = 0
            
        # Read the best frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("Error: Could not extract the best frame.")
            sys.exit(1)
            
        # Re-run model on the best frame if we didn't save the results properly
        if best_results is not None and best_frame_idx != 0:
            results = best_results
        else:
            results = model(frame, conf=args.conf, imgsz=1280, device=device)

    # Custom beautiful drawing
    print("Drawing beautiful annotations...")
    res = results[0]
    if len(res.boxes) > 0:
        annotated_frame = draw_beautiful_detections(frame, res.boxes)
    else:
        annotated_frame = frame
        
    cv2.imwrite(args.output, annotated_frame)
    print(f"Success! Annotated image saved to: {args.output}")

if __name__ == "__main__":
    main()
