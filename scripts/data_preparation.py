import os
import glob
import shutil
import cv2
from pathlib import Path

def clean_dataset(image_dir, label_dir):
    """
    Kiểm tra và xóa bỏ các ảnh bị lỗi không đọc được hoặc các file nhãn bị rỗng.
    """
    print(f"\n--- Bắt đầu làm sạch dữ liệu trong: {os.path.basename(image_dir)} ---")
    corrupt_images = 0
    empty_labels = 0
    
    img_paths = glob.glob(os.path.join(image_dir, "*"))
    for img_path in img_paths:
        # 1. Kiểm tra ảnh lỗi không đọc được
        img = cv2.imread(img_path)
        if img is None:
            if os.path.exists(img_path):
                os.remove(img_path)
            corrupt_images += 1
            # Xóa file nhãn tương ứng nếu có
            lbl_path = os.path.join(label_dir, Path(img_path).stem + ".txt")
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            continue
            
        # 2. Kiểm tra file nhãn tương ứng
        lbl_path = os.path.join(label_dir, Path(img_path).stem + ".txt")
        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            if os.path.exists(lbl_path): 
                os.remove(lbl_path)
            if os.path.exists(img_path):
                os.remove(img_path)
            empty_labels += 1
            
    print(f"✔️ Đã loại bỏ {corrupt_images} ảnh bị hỏng.")
    print(f"✔️ Đã loại bỏ {empty_labels} ảnh không có nhãn hoặc nhãn rỗng.")

def merge_and_create_dataset(raw_dir, output_dir):
    """
    Gộp 3 tập dữ liệu (ball, player, pitch) thành 1 tập dữ liệu duy nhất
    với hệ thống nhãn đồng nhất (5 lớp):
    0: ball, 1: goalkeeper, 2: player, 3: referee, 4: pitch
    """
    # Xóa thư mục cũ nếu có để làm sạch dữ liệu lỗi trước đó
    if os.path.exists(output_dir):
        print(f"Đang xóa thư mục dữ liệu cũ tại: {output_dir}")
        shutil.rmtree(output_dir)
        
    print(f"Đang tạo cấu trúc thư mục YOLO mới tại: {output_dir}")
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    
    # CẬP NHẬT OFFSET CHUẨN:
    # 1. ball: 0: ball -> offset = 0
    # 2. player: 0: ball, 1: goalkeeper, 2: player, 3: referee -> offset = 0
    # 3. pitch: 0: pitch -> chuyển thành class 4 -> offset = 4
    datasets = {
        "ball": {"path": os.path.join(raw_dir, "ball"), "offset": 0},
        "player": {"path": os.path.join(raw_dir, "player"), "offset": 0},
        "pitch": {"path": os.path.join(raw_dir, "pitch"), "offset": 4}
    }
    
    for ds_name, config in datasets.items():
        print(f"\n--- Đang xử lý và gộp dữ liệu từ dataset: {ds_name.upper()} ---")
        if not os.path.exists(config["path"]):
            print(f"⚠️ Cảnh báo: Thư mục dataset '{ds_name}' không tồn tại tại {config['path']}. Bỏ qua...")
            continue
            
        for split in splits:
            img_dir = os.path.join(config["path"], split, "images")
            lbl_dir = os.path.join(config["path"], split, "labels")
            
            if not os.path.exists(img_dir): 
                continue
                
            img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
            count = 0
            for img_path in img_files:
                base_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(base_name)[0]
                lbl_path = os.path.join(lbl_dir, f"{name_without_ext}.txt")
                
                if os.path.exists(lbl_path):
                    target_lbl_path = os.path.join(output_dir, "labels", split, f"{name_without_ext}.txt")
                    
                    # Đọc và ghi nhãn với offset tương ứng
                    # Sử dụng chế độ 'a' (append) để gộp nhãn nếu ảnh đã tồn tại trước đó
                    with open(lbl_path, 'r') as f_in, open(target_lbl_path, 'a') as f_out:
                        for line in f_in.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_cls = int(parts[0])
                                new_cls = old_cls + config["offset"]
                                # Đảm bảo lớp pitch (offset 4) hoặc các lớp khác không bị tràn lớp (max là 4)
                                if new_cls <= 4:
                                    f_out.write(f"{new_cls} {' '.join(parts[1:])}\n")
                    
                    # Copy ảnh sang thư mục đích
                    target_img_path = os.path.join(output_dir, "images", split, base_name)
                    shutil.copy(img_path, target_img_path)
                    count += 1
            print(f"✔️ Đã gộp thành công {count} ảnh & nhãn cho tập [{split}]")

    # Tiến hành làm sạch dữ liệu sau khi gộp
    for split in splits:
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            clean_dataset(img_dir, lbl_dir)

    print(f"\n🎉 Đã gộp và làm sạch dữ liệu CHUẨN XÁC thành công tại: {output_dir}")

def create_yaml_config(output_dir, project_root):
    """
    Tự động tạo file YAML cấu hình tập dữ liệu đã gộp để train YOLO cục bộ (local).
    """
    yaml_dest = os.path.join(project_root, "data", "dataset_merged_local.yaml")
    yaml_content = f"""# File cấu hình tự động cho tập dữ liệu đã gộp (Chạy Local)
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
  4: pitch
"""
    with open(yaml_dest, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"🎉 Đã tự động tạo file config YOLO cục bộ tại: {yaml_dest}")

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(ROOT_DIR, "data", "raw_datasets")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "merged_dataset")
    
    # Chạy quy trình gộp dữ liệu
    merge_and_create_dataset(RAW_DIR, OUTPUT_DIR)
    
    # Tạo cấu hình YAML
    create_yaml_config(OUTPUT_DIR, ROOT_DIR)

if __name__ == "__main__":
    main()
