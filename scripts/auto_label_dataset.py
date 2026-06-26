import os
import sys
import shutil
import random
import numpy as np

def calculate_iou(box1, box2):
    """
    Tính IoU giữa 2 box để thực hiện NMS loại bỏ trùng lặp.
    box format: [x_center, y_center, width, height] (YOLO format normalized)
    """
    # Chuyển sang [xmin, ymin, xmax, ymax]
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    # Tìm tọa độ giao nhau
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def apply_cross_model_nms(predictions, iou_threshold=0.6):
    """
    NMS chéo giữa các mô hình để tránh trùng lặp box của cùng 1 class.
    predictions: list of dict {'class': int, 'box': [...], 'conf': float}
    """
    if not predictions:
        return []
    
    # Sắp xếp theo confidence giảm dần
    predictions = sorted(predictions, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    while predictions:
        best = predictions.pop(0)
        keep.append(best)
        
        # Lọc ra các dự đoán khác trùng class và có IoU cao với best
        remaining = []
        for pred in predictions:
            if pred['class'] == best['class']:
                iou = calculate_iou(best['box'], pred['box'])
                if iou < iou_threshold:
                    remaining.append(pred)
            else:
                remaining.append(pred)
        predictions = remaining
        
    return keep

def main():
    # 1. Cài đặt/kiểm tra thư viện ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Đang cài đặt thư viện ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO

    # Thiết lập thư mục
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "raw_images")
    OUTPUT_DATASET_DIR = os.path.join(ROOT_DIR, "data", "auto_labeled_dataset")
    
    BALL_MODEL_PATH = os.path.join(ROOT_DIR, "data", "models", "football-ball-detection.pt")
    PLAYER_MODEL_PATH = os.path.join(ROOT_DIR, "data", "models", "football-player-detection.pt")
    PITCH_MODEL_PATH = os.path.join(ROOT_DIR, "data", "models", "football-pitch-detection.pt")

    # Kiểm tra sự tồn tại của ảnh thô và mô hình
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ Thư mục ảnh thô không tồn tại: {RAW_IMAGES_DIR}")
        return

    for path, name in [(BALL_MODEL_PATH, "Ball Model"), (PLAYER_MODEL_PATH, "Player Model"), (PITCH_MODEL_PATH, "Pitch Model")]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy tệp mô hình {name} tại: {path}")
            print("Vui lòng chạy scripts/download_assets.py trước.")
            return

    # Load các mô hình
    print("\n--- ĐANG LOAD CÁC MÔ HÌNH HÌNH ẢNH ---")
    ball_model = YOLO(BALL_MODEL_PATH)
    player_model = YOLO(PLAYER_MODEL_PATH)
    pitch_model = YOLO(PITCH_MODEL_PATH)
    print("🎉 Load thành công 3 mô hình.")

    # Tìm danh sách ảnh thô (.jpg, .jpeg, .png)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(valid_extensions)]
    print(f"Tìm thấy {len(all_images)} ảnh thô trong thư mục raw_images.")

    if not all_images:
        print("❌ Không có ảnh nào để gán nhãn.")
        return

    # Dọn dẹp thư mục đầu ra nếu có
    if os.path.exists(OUTPUT_DATASET_DIR):
        print(f"Dọn dẹp thư mục đích cũ: {OUTPUT_DATASET_DIR}...")
        shutil.rmtree(OUTPUT_DATASET_DIR)
    
    # Tạo cấu trúc thư mục tạm thời để chứa kết quả gán nhãn
    temp_images_dir = os.path.join(OUTPUT_DATASET_DIR, "temp_images")
    temp_labels_dir = os.path.join(OUTPUT_DATASET_DIR, "temp_labels")
    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    print("\n--- BẮT ĐẦU QUÁ TRÌNH GÁN NHÃN TỰ ĐỘNG (CONF >= 0.3) ---")
    
    labeled_count = 0
    skipped_count = 0

    for idx, img_name in enumerate(all_images):
        img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        
        # Danh sách chứa tất cả các detection của ảnh này
        img_predictions = []

        try:
            # 1. Dự đoán từ mô hình Ball (Chỉ lấy quả bóng - class 0)
            ball_res = ball_model(img_path, conf=0.3, verbose=False)[0]
            for box in ball_res.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 0:  # Class 0 của ball model chính là ball
                    xywh = box.xywhn[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].item())
                    img_predictions.append({'class': 0, 'box': xywh, 'conf': conf})

            # 2. Dự đoán từ mô hình Player (Nhận diện ball, goalkeeper, player, referee)
            player_res = player_model(img_path, conf=0.3, verbose=False)[0]
            for box in player_res.boxes:
                cls_id = int(box.cls[0].item())
                # Ánh xạ lớp trực tiếp (0: ball, 1: goalkeeper, 2: player, 3: referee)
                if cls_id in [0, 1, 2, 3]:
                    xywh = box.xywhn[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].item())
                    img_predictions.append({'class': cls_id, 'box': xywh, 'conf': conf})

            # 3. Dự đoán từ mô hình Pitch (Khung sân - Ánh xạ về class 4)
            pitch_res = pitch_model(img_path, conf=0.3, verbose=False)[0]
            for box in pitch_res.boxes:
                # Bất kỳ class nào phát hiện được từ mô hình pitch đều ánh xạ thành 4 - pitch
                xywh = box.xywhn[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())
                img_predictions.append({'class': 4, 'box': xywh, 'conf': conf})

        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh {img_name}: {e}")
            continue

        # Áp dụng NMS chéo để loại bỏ các box trùng nhau (đặc biệt là quả bóng/cầu thủ bị quét trùng)
        final_predictions = apply_cross_model_nms(img_predictions, iou_threshold=0.6)

        # Nếu không nhận diện được đối tượng nào -> Bỏ qua không đưa vào dataset gộp (đúng theo báo cáo)
        if not final_predictions:
            skipped_count += 1
            continue

        # Sao chép ảnh vào thư mục tạm
        shutil.copy(img_path, os.path.join(temp_images_dir, img_name))

        # Lưu file nhãn YOLO tương ứng
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(temp_labels_dir, label_name)
        with open(label_path, "w") as f:
            for pred in final_predictions:
                box_str = " ".join([f"{coord:.6f}" for coord in pred['box']])
                f.write(f"{pred['class']} {box_str}\n")
        
        labeled_count += 1
        if labeled_count % 100 == 0:
            print(f"-> Đã gán nhãn thành công {labeled_count} ảnh...")

    print(f"\n✅ Hoàn thành quét nhãn: {labeled_count} ảnh được gán nhãn, {skipped_count} ảnh trống bị bỏ qua.")

    # 6. Chia tập dữ liệu thành Train và Val (Tỷ lệ 80/20)
    print("\n--- ĐANG CHIA TẬP DỮ LIỆU THÀNH TRAIN VÀ VAL (80/20) ---")
    
    # Tạo cấu trúc thư mục YOLO chuẩn
    train_img_dir = os.path.join(OUTPUT_DATASET_DIR, "images", "train")
    val_img_dir = os.path.join(OUTPUT_DATASET_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, "labels", "train")
    val_lbl_dir = os.path.join(OUTPUT_DATASET_DIR, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Lấy danh sách ảnh đã gán nhãn thành công
    success_images = os.listdir(temp_images_dir)
    random.seed(42)
    random.shuffle(success_images)

    split_idx = int(len(success_images) * 0.8)
    train_images = success_images[:split_idx]
    val_images = success_images[split_idx:]

    # Copy files sang thư mục train
    for img in train_images:
        lbl = os.path.splitext(img)[0] + ".txt"
        shutil.move(os.path.join(temp_images_dir, img), os.path.join(train_img_dir, img))
        shutil.move(os.path.join(temp_labels_dir, lbl), os.path.join(train_lbl_dir, lbl))

    # Copy files sang thư mục val
    for img in val_images:
        lbl = os.path.splitext(img)[0] + ".txt"
        shutil.move(os.path.join(temp_images_dir, img), os.path.join(val_img_dir, img))
        shutil.move(os.path.join(temp_labels_dir, lbl), os.path.join(val_lbl_dir, lbl))

    # Xóa thư mục tạm thời
    shutil.rmtree(temp_images_dir)
    shutil.rmtree(temp_labels_dir)

    print(f"📊 Kết quả chia tập:")
    print(f"   - Tập TRAIN: {len(train_images)} ảnh")
    print(f"   - Tập VAL  : {len(val_images)} ảnh")

    # 7. Sinh file dataset_auto_labeled.yaml cục bộ
    yaml_content = f"""# Tập dữ liệu bóng đá gộp tự động bằng mô hình (Auto-labeled)
path: ../data/auto_labeled_dataset
train: images/train
val: images/val

# Khai báo 5 lớp đối tượng gộp
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
  4: pitch
"""
    yaml_path = os.path.join(ROOT_DIR, "data", "dataset_auto_labeled.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n🎉 HOÀN THÀNH TẠO DATASET AUTO-LABELED!")
    print(f"📍 Tập dữ liệu lưu tại: {OUTPUT_DATASET_DIR}")
    print(f"📍 File cấu hình YOLO lưu tại: {yaml_path}")

if __name__ == "__main__":
    main()
