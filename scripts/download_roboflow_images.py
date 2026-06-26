import os
import sys
import shutil

def main():
    # 1. Cài đặt/kiểm tra thư viện roboflow
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow library is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    # 2. Khởi tạo kết nối Roboflow API
    # Tìm API Key từ Environment Variable hoặc từ file .env
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        env_path = os.path.join(ROOT_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("ROBOFLOW_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
                        
    if not api_key:
        print("⚠️ Không tìm thấy API Key của Roboflow!")
        print("Bạn có thể cấu hình bằng 1 trong 2 cách sau:")
        print("Cách 1: Tạo file `.env` ở thư mục gốc dự án với nội dung: ROBOFLOW_API_KEY=your_api_key")
        print("Cách 2: Set biến môi trường: set ROBOFLOW_API_KEY=your_api_key")
        print("-" * 50)
        api_key = input("Nhập mã Roboflow API Key của bạn để tiếp tục: ").strip()
        if not api_key:
            print("❌ Lỗi: API Key không hợp lệ. Hủy quá trình tải.")
            sys.exit(1)
            
    rf = Roboflow(api_key=api_key)
    workspace_name = "roboflow-jvuqo"

    # Định nghĩa cấu trúc thư mục lưu trữ cục bộ
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATASETS_DIR = os.path.join(ROOT_DIR, "data", "raw_datasets")
    os.makedirs(RAW_DATASETS_DIR, exist_ok=True)

    # 3. Định nghĩa các tập dữ liệu cần tải về
    datasets = {
        "ball": {
            "project": "football-ball-detection-rejhg",
            "version": 2,
            "dest": os.path.join(RAW_DATASETS_DIR, "ball")
        },
        "player": {
            "project": "football-players-detection-3zvbc",
            "version": 10,
            "dest": os.path.join(RAW_DATASETS_DIR, "player")
        },
        "pitch": {
            "project": "football-field-detection-f07vi",
            "version": 12,
            "dest": os.path.join(RAW_DATASETS_DIR, "pitch")
        }
    }

    # 4. Tải các dataset từ Roboflow
    for name, config in datasets.items():
        print(f"\n=== ĐANG TẢI DATASET: {name.upper()} (Project: {config['project']}, Version: {config['version']}) ===")
        
        # Xóa thư mục cũ nếu có để tránh dữ liệu bị lỗi/trùng lặp
        if os.path.exists(config["dest"]):
            print(f"Thư mục cũ đã tồn tại tại {config['dest']}. Đang dọn dẹp để tải mới...")
            shutil.rmtree(config["dest"])
            
        try:
            project = rf.workspace(workspace_name).project(config["project"])
            dataset = project.version(config["version"]).download(
                model_format="yolov8",
                location=config["dest"]
            )
            print(f"🎉 Đã tải thành công dataset {name.upper()} về thư mục: {config['dest']}")
        except Exception as e:
            print(f"❌ Lỗi khi tải dataset {name.upper()}: {e}")

    print("\n🎉 HOÀN THÀNH TẢI TOÀN BỘ 3 DATASETS TỪ ROBOFLOW!")
    print(f"Dữ liệu thô đã được lưu tại: {RAW_DATASETS_DIR}")

if __name__ == "__main__":
    main()
