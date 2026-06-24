import os
import sys
import shutil
import json
import time

# Adjust path to import from src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from football_ai.config.loader import load_config

def evaluate_run(output_dir):
    """
    Parses the analytics_tracks.json of a run to compute a detection quality score.
    """
    tracks_path = os.path.join(output_dir, "analytics_tracks.json")
    if not os.path.exists(tracks_path):
        # Fallback to check modular tracking output if running in modular mode
        tracks_path = os.path.join(output_dir, "tracking", "analytics_tracks.json")
        
    if not os.path.exists(tracks_path):
        return None, "No tracking output found"
        
    try:
        with open(tracks_path, "r", encoding="utf-8") as f:
            tracks = json.load(f)
    except Exception as e:
        return None, f"Failed to read track file: {e}"
        
    if not tracks:
        return {
            "score": 0.0, "ball": 0, "referee": 0, "goalkeeper": 0, "player": 0, "avg_conf": 0.0
        }, "Empty tracking data"

    # Count detections by role
    ball_count = sum(1 for t in tracks if t.get("role") == "ball")
    referee_count = sum(1 for t in tracks if t.get("role") == "referee")
    gk_count = sum(1 for t in tracks if t.get("role") == "goalkeeper")
    player_count = sum(1 for t in tracks if t.get("role") == "player")
    
    total_conf = sum(t.get("confidence", 0.0) for t in tracks)
    avg_conf = total_conf / len(tracks)
    
    raw_score = (ball_count * 5.0) + (referee_count * 2.0) + (gk_count * 2.0) + (player_count * 1.0)
    final_score = raw_score * avg_conf
    
    stats = {
        "score": round(final_score, 2),
        "ball": ball_count,
        "referee": referee_count,
        "goalkeeper": gk_count,
        "player": player_count,
        "avg_conf": round(avg_conf, 3)
    }
    return stats, None

def run_single_pipeline(config_path, input_path, output_dir, mode, device, max_frames):
    from football_ai.pipelines.analytics_pipeline import AnalyticsPipeline
    config = load_config(config_path)
    config.video.input_path = input_path
    config.video.output_dir = output_dir
    config.detection.mode = mode
    config.device = device
    
    os.makedirs(output_dir, exist_ok=True)
    pipeline = AnalyticsPipeline(config)
    pipeline.run(max_frames=max_frames)

def run_tracking(config, input_path, output_dir, max_frames, mode):
    from football_ai.pipelines.tracking_pipeline import TrackingPipeline
    track_output = os.path.join(output_dir, "tracking")
    os.makedirs(track_output, exist_ok=True)
    config.video.input_path = input_path
    config.video.output_dir = track_output
    config.detection.mode = mode
    pipeline = TrackingPipeline(config)
    pipeline.run(max_frames=max_frames)

def run_classification(config, input_path, output_dir, max_frames, mode):
    from football_ai.pipelines.classification_pipeline import ClassificationPipeline
    class_output = os.path.join(output_dir, "classification")
    os.makedirs(class_output, exist_ok=True)
    config.video.input_path = input_path
    config.video.output_dir = class_output
    config.detection.mode = mode
    pipeline = ClassificationPipeline(config)
    pipeline.run(max_frames=max_frames)

def run_modular_flow(config_path, input_path, output_dir, mode, device, max_frames):
    config = load_config(config_path)
    config.device = device
    
    print(f"\n>>> Đang chạy mô-đun 1: Phát hiện đối tượng + Theo dõi chuyển động (Detect+Tracking) [{mode}]...")
    t_track_start = time.time()
    run_tracking(config, input_path, output_dir, max_frames, mode)
    t_track_elapsed = time.time() - t_track_start
    
    print(f"\n>>> Đang chạy mô-đun 2: Phân loại đội hình (Classification) [{mode}]...")
    t_class_start = time.time()
    run_classification(config, input_path, output_dir, max_frames, mode)
    t_class_elapsed = time.time() - t_class_start
    
    return t_track_elapsed, t_class_elapsed

def main():
    print("="*60)
    print("⚽  HỆ THỐNG TRÍ TUỆ NHÂN TẠO PHÂN TÍCH BÓNG ĐÁ  ⚽")
    print("="*60)
    
    # 1. Nhập đường dẫn Video đầu vào
    default_input = "data/raw/sample_10s.mp4"
    input_path = input("🎥 Nhập đường dẫn Video phân tích [Mặc định: data/raw/sample_10s.mp4]: ").strip()
    if not input_path:
        input_path = default_input
        
    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy tệp video tại '{input_path}'!")
        sys.exit(1)
        
    # Tự động sinh tên thư mục kết quả dựa trên tên video
    video_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join("data", "outputs", video_filename)
    print(f"📂 Thư mục kết quả tự động: {output_dir}")
    
    # 2. Chọn Chế độ Chạy (Pipeline Mode)
    print("\n" + "="*50)
    print("📋 MENU 1: CHỌN CHẾ ĐỘ XỬ LÝ VIDEO")
    print(" [1] Chạy Full Pipeline (Tổng hợp: Phân tích + Sự kiện + Bản đồ Radar)")
    print(" [2] Chạy từng Mô-đun lẻ (Tạo video Tracking & Classification riêng biệt)")
    print("="*50)
    
    mode_choice = input("Nhập chế độ chạy (1 hoặc 2) [Mặc định: 1]: ").strip()
    if not mode_choice:
        mode_choice = "1"
        
    # 3. Chọn Cấu hình Mô hình (Model Configuration)
    print("\n" + "="*50)
    print("🤖 MENU 2: CHỌN CẤU HÌNH MÔ HÌNH NHẬN DIỆN")
    print(" [1] Chạy bằng 1-Model gộp (best.pt) - [Tối ưu tốc độ, tiết kiệm RAM/VRAM]")
    print(" [2] Chạy bằng 3-Model riêng biệt (Player, Ball, Pitch) - [Độ chính xác chi tiết]")
    print(" [3] Chạy song song cả hai và tự động chọn mô hình ít sai số nhất (So sánh & Đánh giá)")
    print("="*50)
    
    model_choice = input("Nhập cấu hình mô hình (1, 2 hoặc 3) [Mặc định: 1]: ").strip()
    if not model_choice:
        model_choice = "1"

    # 4. Chọn Thiết bị chạy (Hardware Device)
    print("\n" + "="*50)
    print("🖥️  MENU 3: CHỌN PHẦN CỨNG XỬ LÝ")
    print(" [1] Chạy bằng GPU (CUDA) - Đề xuất cho hiệu năng tốt nhất")
    print(" [2] Chạy bằng CPU - Phù hợp với máy không có card đồ họa rời")
    print("="*50)
    
    device_choice = input("Nhập thiết bị xử lý (1 hoặc 2) [Mặc định: 1]: ").strip()
    
    import torch
    if device_choice == "2":
        device = "cpu"
    else:
        device = "cuda"
        
    if device == "cuda" and not torch.cuda.is_available():
        print("\n⚠️  Cảnh báo: CUDA không khả dụng trên hệ thống! Tự động chuyển đổi sang CPU.")
        device = "cpu"
    else:
        print(f"\n✅ Đã chọn phần cứng: {device.upper()}\n")

    # 5. Nhập số frame chạy thử
    max_frames = 150
    frames_input = input("⏱️  Nhập số frame muốn chạy (-1 để chạy hết cả video) [Mặc định: 150]: ").strip()
    if frames_input:
        try:
            max_frames = int(frames_input)
        except ValueError:
            pass
            
    start_time = time.time()
    config_path = "configs/merge_test.yaml"
    
    # Tính toán tổng số frame trong video để đo đạc FPS chuẩn
    import cv2
    cap = cv2.VideoCapture(input_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    frames_run = max_frames if (0 < max_frames < total_video_frames) else total_video_frames
    
    # Biến lưu trữ thời gian chạy và trạng thái để báo cáo cuối
    t_elapsed = 0.0
    
    if model_choice == "1":
        # === CHẠY BẢN 1-MODEL GỘP ===
        t_start = time.time()
        if mode_choice == "1":
            run_single_pipeline(config_path, input_path, output_dir, "merged_model", device, max_frames)
        else:
            t_tr, t_cl = run_modular_flow(config_path, input_path, output_dir, "merged_model", device, max_frames)
        t_elapsed = time.time() - t_start
        fps = frames_run / t_elapsed if t_elapsed > 0 else 0
        
        print(f"\n🎉 Hoàn thành chạy bằng 1-Model gộp!")
        print(f"📂 Kết quả được lưu tại: {output_dir}")
        print(f"⏱️  Thời gian xử lý: {t_elapsed:.2f} giây")
        print(f"⚡ Tốc độ xử lý: {fps:.2f} FPS")

    elif model_choice == "2":
        # === CHẠY BẢN 3-MODEL RIÊNG BIỆT ===
        t_start = time.time()
        if mode_choice == "1":
            run_single_pipeline(config_path, input_path, output_dir, "three_models", device, max_frames)
        else:
            t_tr, t_cl = run_modular_flow(config_path, input_path, output_dir, "three_models", device, max_frames)
        t_elapsed = time.time() - t_start
        fps = frames_run / t_elapsed if t_elapsed > 0 else 0
        
        print(f"\n🎉 Hoàn thành chạy bằng 3-Model riêng biệt!")
        print(f"📂 Kết quả được lưu tại: {output_dir}")
        print(f"⏱️  Thời gian xử lý: {t_elapsed:.2f} giây")
        print(f"⚡ Tốc độ xử lý: {fps:.2f} FPS")

    elif model_choice == "3":
        # === CHẠY SONG SONG CẢ HAI ĐỂ SO SÁNH & TỰ ĐỘNG CHỌN BẢN TỐT NHẤT ===
        temp_three_dir = os.path.join(ROOT, "data", "outputs", "temp_three")
        temp_merged_dir = os.path.join(ROOT, "data", "outputs", "temp_merged")
        
        # 1. Chạy 3-Model
        print("\n>>> Đang chạy thử nghiệm Mô hình 3-Model...")
        t_three_start = time.time()
        if mode_choice == "1":
            run_single_pipeline(config_path, input_path, temp_three_dir, "three_models", device, max_frames)
        else:
            run_modular_flow(config_path, input_path, temp_three_dir, "three_models", device, max_frames)
        t_three_elapsed = time.time() - t_three_start
        stats_three, err_three = evaluate_run(temp_three_dir)
        
        # 2. Chạy 1-Model gộp
        print("\n>>> Đang chạy thử nghiệm Mô hình 1-Model Gộp...")
        t_merged_start = time.time()
        if mode_choice == "1":
            run_single_pipeline(config_path, input_path, temp_merged_dir, "merged_model", device, max_frames)
        else:
            run_modular_flow(config_path, input_path, temp_merged_dir, "merged_model", device, max_frames)
        t_merged_elapsed = time.time() - t_merged_start
        stats_merged, err_merged = evaluate_run(temp_merged_dir)
        
        # 3. Xuất báo cáo so sánh chất lượng nhận diện (Ít sai số nhất)
        print("\n" + "="*60)
        print("          BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH")
        print("="*60)
        if not err_three:
            fps_three = frames_run / t_three_elapsed if t_three_elapsed > 0 else 0
            print(f"1. Cấu hình 3-Model riêng biệt:")
            print(f"   - Số lần phát hiện bóng:      {stats_three['ball']}")
            print(f"   - Số lần phát hiện trọng tài:  {stats_three['referee']}")
            print(f"   - Số lần phát hiện cầu thủ:   {stats_three['player']}")
            print(f"   - Độ tự tin trung bình:       {stats_three['avg_conf'] * 100:.1f}%")
            print(f"   - Thời gian xử lý:            {t_three_elapsed:.2f} giây ({fps_three:.2f} FPS)")
            print(f"   * ĐIỂM SỐ NHẬN DIỆN CHUẨN:    {stats_three['score']}")
        else:
            print(f"❌ Lỗi chạy 3-Model: {err_three}")
            
        print("-"*60)
        if not err_merged:
            fps_merged = frames_run / t_merged_elapsed if t_merged_elapsed > 0 else 0
            print(f"2. Cấu hình 1-Model gộp (best.pt):")
            print(f"   - Số lần phát hiện bóng:      {stats_merged['ball']}")
            print(f"   - Số lần phát hiện trọng tài:  {stats_merged['referee']}")
            print(f"   - Số lần phát hiện cầu thủ:   {stats_merged['player']}")
            print(f"   - Độ tự tin trung bình:       {stats_merged['avg_conf'] * 100:.1f}%")
            print(f"   - Thời gian xử lý:            {t_merged_elapsed:.2f} giây ({fps_merged:.2f} FPS)")
            print(f"   * ĐIỂM SỐ NHẬN DIỆN CHUẨN:    {stats_merged['score']}")
        else:
            print(f"❌ Lỗi chạy 1-Model gộp: {err_merged}")
        print("="*60)
        
        # 4. Báo cáo so sánh thời gian chạy thực tế
        if not err_three and not err_merged:
            speedup = t_three_elapsed / t_merged_elapsed if t_merged_elapsed > 0 else 0.0
            time_saved = ((t_three_elapsed - t_merged_elapsed) / t_three_elapsed) * 100 if t_three_elapsed > 0 else 0.0
            print("\n" + "="*60)
            print("          BÁO CÁO SO SÁNH TỐC ĐỘ XỬ LÝ (SPEED COMPARISON)")
            print("="*60)
            print(f" - Cấu hình 3-Model:      {t_three_elapsed:.2f}s ({fps_three:.2f} FPS)")
            print(f" - Cấu hình 1-Model gộp:  {t_merged_elapsed:.2f}s ({fps_merged:.2f} FPS)")
            print(f" ⚡ HIỆU SUẤT TĂNG TỐC:   {speedup:.2f}x nhanh hơn ({time_saved:.1f}% giảm độ trễ)")
            print("="*60)
        
        # 5. Quyết định tự động lựa chọn bản có sai số thấp nhất (điểm cao nhất)
        if err_three and err_merged:
            print("❌ Lỗi: Cả 2 mô hình chạy thử nghiệm đều thất bại!")
            sys.exit(1)
        elif err_three:
            winner_mode, winner_dir = "Merged-Model", temp_merged_dir
            reason = "Bản 3-Model gặp lỗi trong quá trình xử lý"
        elif err_merged:
            winner_mode, winner_dir = "Three-Model", temp_three_dir
            reason = "Bản 1-Model gộp gặp lỗi trong quá trình xử lý"
        else:
            # Điểm chất lượng cao hơn đồng nghĩa nhận diện chuẩn xác nhất (ít bỏ lỡ bóng/cầu thủ/trọng tài)
            if stats_merged["score"] >= stats_three["score"]:
                winner_mode, winner_dir = "Merged-Model", temp_merged_dir
                reason = f"Đạt điểm chất lượng nhận diện cao hơn ({stats_merged['score']} so với {stats_three['score']})"
            else:
                winner_mode, winner_dir = "Three-Model", temp_three_dir
                reason = f"Đạt điểm chất lượng nhận diện cao hơn ({stats_three['score']} so với {stats_merged['score']})"
                
        print(f"\n🏆 MÔ HÌNH CHIẾN THẮNG (TỐI ƯU NHẤT): {winner_mode}")
        print(f"👉 Lý do chọn: {reason}")
        print(f"📦 Đang đóng gói kết quả tốt nhất vào thư mục: {output_dir}")
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(winner_dir, output_dir)
        
        # Dọn dẹp các tệp tạm thời
        print("🧹 Đang dọn dẹp các tệp tạm thời...")
        for t_dir in [temp_three_dir, temp_merged_dir]:
            if os.path.exists(t_dir):
                shutil.rmtree(t_dir)
                
        t_elapsed = time.time() - start_time
        print(f"\n🎉 Hoàn thành so sánh! Kết quả tối ưu nhất đã được lưu tại: {output_dir}")

    total_time = time.time() - start_time
    print(f"\n⏱️  Tổng thời gian xử lý toàn bộ tiến trình: {total_time:.2f} giây.")
    print(f"💡 Để xem kết quả trên giao diện web, hãy khởi chạy Streamlit:")
    print(f"   streamlit run src/football_ai/dashboard.py")

if __name__ == "__main__":
    main()
