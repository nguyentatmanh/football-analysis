# ⚽ Football AI Video Analysis & Commentary System

Hệ thống phân tích video bóng đá tự động, tích hợp trí tuệ nhân tạo (AI) toàn diện: từ phát hiện đối tượng, theo dõi (tracking), định vị sân bóng (homography), đo đạc chỉ số vật lý, tạo bản đồ nhiệt (heatmap), đến tự động nhận diện sự kiện trận đấu và bình luận bằng giọng nói tiếng Việt/Anh sinh động bằng LLM & TTS.

---

## 🚀 Các Tính Năng Chính (Key Features)

*   **Nhận diện 5 lớp nâng cao (YOLOv8x Merged Model):** Phát hiện chính xác Cầu thủ (Player), Trọng tài (Referee), Thủ môn (Goalkeeper), Quả bóng (Ball) và Đường biên sân bóng (Pitch Keypoints) bằng mô hình gộp tối ưu hiệu năng.
*   **Theo dõi đa đối tượng (ByteTrack):** Liên kết vết di chuyển của cầu thủ và bóng ổn định qua các khung hình, lọc nhiễu quỹ đạo bằng EMA.
*   **Phân loại đội bóng thông minh (Unsupervised Team Clustering):** Tự động phân tích màu sắc trang phục (kit colors) để chia nhóm cầu thủ thành 2 đội mà không cần huấn luyện trước.
*   **Tọa độ hóa góc nhìn Camera (Homography Pitch Mapping):** Ánh xạ tọa độ 2D của cầu thủ từ góc quay camera lên sơ đồ 2D trực quan (Tactical Radar).
*   **Phân tích di chuyển & Bản đồ nhiệt (Heatmaps & Analytics):** Tính toán quãng đường di chuyển (m), tốc độ trung bình, tốc độ cực đại (km/h) của từng cầu thủ; vẽ bản đồ mật độ hoạt động (Heatmap) dạng PNG chất lượng cao.
*   **Nhận diện sự kiện & Tranh chấp bóng:** Tự động bắt sự kiện chuyền bóng, sút bóng, và kiểm soát bóng (Possession) dựa trên khoảng cách vật lý từ chân cầu thủ tới bóng.
*   **Bình luận viên AI (LLM & TTS Commentary):** Tích hợp mô hình ngôn ngữ lớn (Llama-3) nâng cao chất lượng câu thoại, kết hợp Edge-TTS để đọc bình luận theo thời gian thực của video trận đấu.
*   **Bảng điều khiển Premium (Streamlit Dashboard):** Giao diện Web hiển thị video kết quả phát mượt mà, sơ đồ radar chạy song song, biểu đồ kiểm soát bóng và bảng thống kê chỉ số vật lý của cầu thủ.

---

## 🛠️ Hướng dẫn cài đặt (Installation)

### 1. Tạo môi trường ảo và cài đặt thư viện
**Trên Windows:**
Chạy file cài đặt tự động môi trường (tạo Virtualenv và cài đặt dependencies qua `pip`):
```cmd
scripts\setup_env.bat
```

**Trên Linux / macOS:**
```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 2. Tải mô hình weights và video mẫu
Chạy script tải tự động các mô hình YOLO (bao gồm bản gộp `best.pt` và các bản đơn lẻ) từ Google Drive cùng video demo:
```cmd
python scripts/download_assets.py
```

---

## 🎬 Hướng dẫn vận hành hệ thống (Usage Guide)

Hệ thống hỗ trợ 2 công cụ chính để chạy phân tích và hiển thị kết quả:

### BƯỚC 1: Chạy phân tích video bằng CLI Master Script
Chúng ta sử dụng Master Script **[run_pipeline.py](scripts/run_pipeline.py)** với giao diện tương tác trực tiếp bằng phím bấm cực kỳ tiện lợi:

```cmd
python scripts/run_pipeline.py
```

**Các lựa chọn khi chạy:**
1.  **Đường dẫn Video:** Nhập đường dẫn video (ví dụ: `data/raw/sample.mp4`). Kết quả sẽ được lưu vào thư mục tự động đặt tên theo video là `data/outputs/[tên_video]`.
2.  **Chọn chế độ chạy (Pipeline Mode):**
    *   **Phím 1:** Chạy Full Pipeline (Tạo video tổng hợp phân tích + phát hiện sự kiện + ghép tiếng bình luận).
    *   **Phím 2:** Chạy mô-đun lẻ (Tạo các video Tracking và video Classification riêng biệt để bạn chuyển đổi xem trên giao diện web).
3.  **Chọn cấu hình mô hình (Model Choice):**
    *   **Phím 1:** Chạy bằng 1-Model gộp (`best.pt`) - Khuyến nghị cho tốc độ nhanh và tiết kiệm VRAM.
    *   **Phím 2:** Chạy bằng 3-Model riêng lẻ.
    *   **Phím 3:** Chạy cả 2 cấu hình và tự động so sánh tốc độ, độ chính xác rồi chọn mô hình tối ưu nhất.
4.  **Lựa chọn phần cứng (Compute Device):**
    *   **Phím 1:** Chạy bằng GPU (CUDA) - Khuyến nghị để đạt hiệu năng xử lý cao nhất.
    *   **Phím 2:** Chạy bằng CPU.
5.  **Nhập số frame chạy thử:** Gõ số frame cần chạy thử nghiệm (ví dụ `150` để test nhanh) hoặc `-1` để chạy toàn bộ video.

---

### BƯỚC 2: Khởi động Streamlit Dashboard để xem kết quả trực quan
Sau khi phân tích xong, khởi động giao diện Web để trình chiếu kết quả:

```cmd
streamlit run src/football_ai/dashboard.py
```

**Cách xem:**
1.  Trình duyệt web sẽ tự động mở tab `http://localhost:8501`.
2.  Tại cột bên trái (Sidebar):
    *   Chọn thư mục chạy (Run Folder) tương ứng với tên video bạn vừa phân tích (ví dụ: `sample_10s` hoặc `test_30s`).
3.  Trải nghiệm các Tab phân tích:
    *   📺 **Video Phân Tích:** Chọn hiển thị video của từng mô-đun hoặc bản Full Pipeline. Video được tối ưu hóa chuẩn H.264 chạy mượt mà ngay trên trình duyệt.
    *   🗺️ **Bản Đồ Nhiệt (Heatmaps):** Hiển thị bản đồ mật độ di chuyển của Đội A, Đội B, Trọng tài, Quả bóng và từng cầu thủ cụ thể.
    *   📊 **Kiểm Soát Bóng:** Biểu đồ hình bánh ngọt so sánh tỷ lệ kiểm soát bóng của hai đội.
    *   🏃 **Thống Kê Di Chuyển:** Bảng xếp hạng quãng đường di chuyển và tốc độ của các cầu thủ cùng biểu đồ so sánh trực quan.

---

## 🧠 Hướng Dẫn Chạy Bình Luận Tự Động (AI Commentary Guide)

### 1. Chi tiết cấu hình File cấu hình (YAML Config)
Để kích hoạt tính năng bình luận tự động bằng mô hình ngôn ngữ lớn (LLM) và tổng hợp giọng nói (TTS), bạn cần cấu hình lại phần `commentary` trong tệp cấu hình (ví dụ: `configs/merge_test.yaml` hoặc `configs/default.yaml`).

Dưới đây là chi tiết các trường cấu hình cần lưu ý:
```yaml
commentary:
  enabled: true                 # [true/false] Bật/Tắt toàn bộ hệ thống xử lý sự kiện & bình luận chữ.
  use_llm_enhancement: true     # [true/false] BẬT để dùng LLM (Llama-3) viết lại câu thoại sinh động. 
                                # (Tắt đi thì chữ bình luận sẽ là các câu mẫu cố định trong code).
  llm_model_name: "meta-llama/Meta-Llama-3-8B-Instruct"  # Tên model trên Hugging Face.
  tts_enabled: true             # [true/false] BẬT để tạo file âm thanh giọng nói và ghép vào video.
  tts_language: "vi"            # Ngôn ngữ đọc thoại: "vi" (Tiếng Việt) hoặc "en" (Tiếng Anh).
  display_duration_seconds: 2.5 # Thời gian hiển thị chữ bình luận trên màn hình (giây).
```

### 2. Các yêu cầu chuẩn bị trước khi chạy LLM (Llama-3)
Vì mô hình **Meta-Llama-3-8B-Instruct** là mô hình đóng có bản quyền của Meta trên Hugging Face, bạn cần thực hiện các bước sau để chạy:
1. **Đăng ký quyền truy cập:** Truy cập [Hugging Face Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), đồng ý với các điều khoản của Meta và đợi Hugging Face phê duyệt (thường mất từ 10 phút đến vài giờ).
2. **Lấy Token cá nhân:** Tạo một Token với quyền `Read` tại [Hugging Face Token Settings](https://huggingface.co/settings/tokens).
3. **Đăng nhập trên thiết bị chạy:**
   * Mở Terminal/CMD trong môi trường ảo `.venv` và chạy lệnh:
     ```bash
     huggingface-cli login
     ```
   * Dán Token của bạn vào và nhấn Enter để xác thực.

### 3. Hạn chế về phần cứng cục bộ (Local Hardware Constraints)
* **Yêu cầu GPU VRAM:** Việc chạy mô hình Llama-3-8B cục bộ yêu cầu card đồ họa có bộ nhớ VRAM tối thiểu **8 GB** đến **12 GB** (khi chạy bản lượng tử hóa 4-bit). Các dòng laptop cá nhân thông thường (VRAM 4 GB hoặc 6 GB như RTX 3050/1650) sẽ gặp lỗi thiếu bộ nhớ (**Out of Memory**) hoặc chạy cực kỳ chậm (độ trễ sinh từ rất lâu).
* **Vấn đề chất lượng bình luận:** Bản bình luận tự động hiện tại đôi lúc sinh câu thoại chưa thực sự tự nhiên như BLV chuyên nghiệp, hoặc có thể bị nhầm lẫn/lệch sự kiện (ví dụ: bóng đã chuyền đi nhưng LLM vẫn bình luận về tranh chấp trước đó) do độ trễ xử lý sự kiện tích lũy qua các khung hình.

### 4. Hướng dẫn chạy trên nền tảng đám mây (Google Colab / Kaggle)
Để khắc phục giới hạn phần cứng laptop cá nhân, dự án đã cung cấp sẵn file Jupyter Notebook **[Cloud_Inference.ipynb](Cloud_Inference.ipynb)** được tối ưu hóa để chạy trực tiếp trên GPU miễn phí (NVIDIA T4 16GB) của Colab hoặc Kaggle:

1. Nén thư mục mã nguồn dự án thành định dạng `.zip` (loại bỏ thư mục `data/outputs/` và các file video nặng để giảm dung lượng tải lên).
2. Tải file `.zip` và video đầu vào lên thư mục làm việc của Google Colab hoặc Kaggle.
3. Mở file **`Cloud_Inference.ipynb`** và tiến hành chạy tuần tự các bước:
   * Giải nén mã nguồn dự án.
   * Cài đặt thư viện dependencies cần thiết bằng lệnh:
     ```bash
     !pip install ultralytics gtts transformers accelerate bitsandbytes huggingface_hub
     ```
   * Đăng nhập tài khoản Hugging Face bằng cách điền Token vào ô nhập của Notebook.
   * Chạy câu lệnh Python CLI phân tích video để tự động xuất ra video kèm audio bình luận tiếng Việt/Anh hoàn chỉnh:
     ```bash
     !python src/football_ai/cli.py --mode analytics --input "data/raw/sample.mp4" --device cuda
     ```
4. Tải tệp video kết quả tại thư mục `data/outputs/` về máy cá nhân của bạn để xem và trình chiếu.

---

## ⚠️ Khuyết Điểm & Hạn Chế Hiện Tại Của Hệ Thống (Known Issues & Limitations)

Mặc dù hệ thống đạt độ hoàn thiện cao, tuy nhiên do giới hạn công nghệ của các mô hình nền tảng, hệ thống vẫn tồn tại một số sai sót kỹ thuật sau:

1.  **Sai số sơ đồ chiến thuật (Tactical Radar Mapping):** 
    *   Tọa độ trên sơ đồ Radar phụ thuộc vào việc tính toán ma trận Homography từ mô hình phát hiện keypoint sân bóng (`football-pitch-detection.pt`). Khi góc quay camera cận cảnh (zoomed-in), camera di chuyển quá nhanh hoặc đường biên bị che khuất nhiều, mô hình keypoint sẽ phát hiện sai lệch dẫn đến tọa độ cầu thủ trên Radar bị nhảy lệch hoặc bay ra ngoài sân.
2.  **Nhầm lẫn vai trò (Classification Mismatch):**
    *   Đôi lúc trọng tài bị nhận diện nhầm thành cầu thủ hoặc ngược lại, đặc biệt là khi màu áo của trọng tài trùng hợp hoặc gần giống với màu áo thi đấu của một trong hai đội bóng.
3.  **Nhảy số ID cầu thủ (ID Fragmentation & Tracking Jumps):**
    *   Mô hình ByteTrack có thể bị mất dấu và cấp một ID mới (nhảy số ID) cho cầu thủ trong các trường hợp: cầu thủ che lấp nhau quá lâu (tranh chấp đông người), cầu thủ di chuyển hoàn toàn ra ngoài khung hình/đường biên rồi quay lại sân.
4.  **Sai lệch đo lường tốc độ (Speed Measurement Errors):**
    *   Tốc độ tức thời (km/h) của cầu thủ tính toán dựa trên sự dịch chuyển tọa độ 2D thực tế qua Homography. Sự rung lắc của camera hoặc sai lệch nhỏ về keypoint sân bóng sẽ phóng đại sai số dịch chuyển, dẫn đến chỉ số tốc độ đôi khi bị vọt cao bất thường (spikes).
5.  **Hạn chế với trận đấu nhịp độ nhanh (High-Tempo Matches):**
    *   Độ chính xác của toàn hệ thống sẽ giảm đi rõ rệt trong các trận đấu có tốc độ luân chuyển bóng cực nhanh, có nhiều pha phản công chớp nhoáng hoặc các pha quay chậm (slow-motion) lặp lại liên tục làm gián đoạn dòng thời gian thực tế.

---

## 📁 Cấu trúc Thư mục Dự án (Project Structure)

```text
├── configs/                  # File cấu hình YAML hệ thống (default, merge_test)
├── data/
│   ├── models/               # Nơi chứa file weights (.pt) của YOLO
│   ├── raw/                  # Chứa các video gốc đầu vào (.mp4)
│   └── outputs/              # Nơi lưu kết quả phân tích theo tên video
├── scripts/                  # Các kịch bản chạy phân tích và đo đạc
│   ├── run_pipeline.py       # Master script chạy tương tác (GPU/CPU, Full/Lẻ/So sánh)
│   └── download_assets.py    # Script tải weights và video mẫu
├── src/
│   └── football_ai/          # Source code cốt lõi của hệ thống
│       ├── detection/        # Mô-đun phát hiện đối tượng YOLO
│       ├── tracking/         # Mô-đun theo dõi ByteTrack
│       ├── classification/   # Bộ phân loại vai trò và đội bóng
│       ├── field_mapping/    # Ánh xạ Homography 2D Radar
│       ├── analytics/        # Thống kê quãng đường, tốc độ, possession
│       ├── commentary/       # Tạo bình luận LLM và giọng nói TTS
│       └── dashboard.py      # Mã nguồn trang Streamlit Dashboard
└── pyproject.toml            # Khai báo thư viện dependencies của dự án
```

---

## ⚡ Thống Kê Hiệu Năng So Sánh
Chuyển đổi từ kiến trúc **3-Model riêng lẻ** sang **1-Model gộp (best.pt)** mang lại sự cải tiến vượt bậc về hiệu năng hệ thống:

| Metric | 3 Separate Models | 1 Merged Model (best.pt) | Cải Tiến |
| :--- | :---: | :---: | :---: |
| **Inference Speed (FPS)** | ~4.15 FPS | **~12.35 FPS** | **Tăng tốc 2.98x** |
| **Model Loading Time** | 15.2s | **6.8s** | **Giảm 55% thời gian nạp** |
| **Memory / VRAM** | Cao (Tải 3 model cùng lúc) | **Thấp (Chỉ tải 1 model)** | **Tiết kiệm bộ nhớ** |
