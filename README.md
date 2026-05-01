## 1. Cấu trúc thư mục

```text
project_folder/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── sales.csv
│   └── sample_submission.csv
└── outputs/
    ├── submission_lightgbm.csv
    ├── submission_ensemble.csv
    ├── submission_anchor_residual.csv
    ├── submission_recursive_yoy.csv
    ├── submission_baseline.csv
    ├── prediction_components.csv
    ├── feature_importance_anchor_revenue.csv
    ├── feature_importance_anchor_revenue.png
    ├── forecast_revenue.png
    └── forecast_cogs.png
```

Trong đó:

- `main.py`: file code chính để huấn luyện mô hình và tạo submission.
- `requirements.txt`: danh sách thư viện cần cài đặt.
- `data/`: thư mục chứa dữ liệu đầu vào.
- `outputs/`: thư mục lưu kết quả dự báo, biểu đồ và feature importance.

## 2. Cài đặt môi trường

Khuyến nghị dùng Python 3.10 hoặc mới hơn.

### Bước 1: Tạo virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu không chạy được do Execution Policy, dùng:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

và chạy các lệnh bằng `.\.venv\Scripts\python.exe`.

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Bước 2: Cài dependencies

```bash
pip install -r requirements.txt
```

## 3. Cách chạy pipeline

### Chạy mặc định để tạo submission

```bash
python main.py --data-dir data --output-dir outputs
```

Sau khi chạy xong, file chính nên nộp lên Kaggle là:

```text
outputs/submission.csv
```

Script cũng tạo thêm một số file submission khác để tham khảo:

```text
outputs/submission_ensemble.csv
outputs/submission_recursive_yoy.csv
outputs/submission_baseline.csv
```

## 4. Chạy kèm cross-validation

Để chạy rolling time-series validation:

```bash
python main.py --data-dir data --output-dir outputs --run-cv
```

Kết quả validation sẽ được lưu tại:

```text
outputs/cv_results.csv
outputs/cv_summary.csv
```

Cross-validation dùng các mốc train end:

```text
2019-06-30
2020-06-30
2021-06-30
```

và đánh giá trên horizon có độ dài bằng tập test.

## 5. Chạy kèm SHAP explainability

SHAP là tùy chọn. Nếu muốn xuất SHAP summary plot:

```bash
python main.py --data-dir data --output-dir outputs --run-shap
```

Kết quả SHAP nếu chạy thành công:

```text
outputs/shap_summary_anchor_revenue.png
```

Nếu chưa cài `shap`, có thể cài thêm:

```bash
pip install shap
```

hoặc cài sẵn từ `requirements.txt`.

## 6. Các tham số quan trọng

Có thể xem toàn bộ tham số bằng:

```bash
python main.py --help
```

Một số tham số chính:

| Tham số              | Ý nghĩa                                                        |    Mặc định |
| -------------------- | -------------------------------------------------------------- | ----------: |
| `--data-dir`         | Thư mục chứa `sales.csv` và `sample_submission.csv`            |      `data` |
| `--output-dir`       | Thư mục lưu output                                             |   `outputs` |
| `--run-cv`           | Chạy rolling time-series validation                            |         Tắt |
| `--run-shap`         | Tạo SHAP summary plot nếu có thư viện `shap`                   |         Tắt |
| `--n-estimators`     | Số cây LightGBM                                                |      `1400` |
| `--learning-rate`    | Learning rate của LightGBM                                     |     `0.018` |
| `--n-jobs`           | Số core CPU dùng cho LightGBM                                  |        `-1` |
| `--target-mode`      | Cách học target cho recursive model: `direct` hoặc `yoy_ratio` | `yoy_ratio` |
| `--recursive-weight` | Trọng số recursive LightGBM trong ensemble                     |      `0.20` |
| `--anchor-weight`    | Trọng số anchor residual LightGBM trong ensemble               |      `0.45` |
| `--baseline-weight`  | Trọng số seasonal baseline trong ensemble                      |      `0.35` |

Ví dụ chỉnh trọng số ensemble:

```bash
python main.py \
  --data-dir data \
  --output-dir outputs \
  --recursive-weight 0.20 \
  --anchor-weight 0.45 \
  --baseline-weight 0.35
```

Trên Windows PowerShell:

```powershell
python main.py `
  --data-dir data `
  --output-dir outputs `
  --recursive-weight 0.20 `
  --anchor-weight 0.45 `
  --baseline-weight 0.35
```

## 7. Mô tả ngắn gọn pipeline

Pipeline gồm 3 thành phần chính:

### 7.1 Seasonal Naive Baseline

Dự báo dựa trên giá trị cùng kỳ năm trước, có điều chỉnh hệ số tăng trưởng YoY gần nhất.

### 7.2 Recursive YoY LightGBM

Mô hình LightGBM học theo chuỗi thời gian với các đặc trưng:

- Calendar features: năm, tháng, quý, tuần, ngày trong năm, ngày trong tuần.
- Fourier seasonality features.
- Lag features: 28, 56, 91, 182, 364, 365, 366, 730 ngày.
- Rolling statistics: mean, std, min, max, median.
- Tỷ lệ giữa các lag theo mùa vụ.

Ở chế độ mặc định `yoy_ratio`, model học phần chênh lệch log so với lag 365 ngày để giảm lỗi lan truyền khi dự báo dài hạn.

### 7.3 Direct Seasonal-Anchor LightGBM

Mô hình anchor residual dựa vào các giá trị cùng kỳ của 1, 2 và 3 năm trước.  
Sau đó LightGBM học residual trên log scale để điều chỉnh seasonal anchor.

### 7.4 Ensemble

Dự báo cuối cùng là trung bình có trọng số của:

- Recursive YoY LightGBM
- Direct Seasonal-Anchor LightGBM
- Seasonal Naive Baseline

Mặc định:

```text
recursive = 0.20
anchor    = 0.45
baseline  = 0.35
```

## 8. Output chính

Sau khi chạy thành công, các file quan trọng trong `outputs/` gồm:

| File                                    | Mô tả                                 |
| --------------------------------------- | ------------------------------------- |
| `submission.csv`                        | File chính nộp Kaggle     |
| `submission_ensemble.csv`               | Submission từ ensemble                |
| `submission_recursive_yoy.csv`          | Submission từ mô hình recursive YoY   |
| `submission_baseline.csv`               | Submission từ seasonal naive baseline |
| `prediction_components.csv`             | So sánh các thành phần dự báo         |
| `feature_importance_anchor_revenue.csv` | Feature importance của anchor model   |
| `feature_importance_anchor_revenue.png` | Biểu đồ top feature importance        |
| `forecast_revenue.png`                  | Biểu đồ train vs forecast Revenue     |
| `forecast_cogs.png`                     | Biểu đồ train vs forecast COGS        |

## 9. Reproducibility

Script đã đặt random seed cố định:

```python
SEED = 42
```

Ngoài ra, LightGBM cũng dùng `random_state=42`.

Để tái lập kết quả:

1. Dùng cùng phiên bản Python và thư viện trong `requirements.txt`.
2. Không thay đổi dữ liệu đầu vào.
3. Chạy cùng command.
4. Giữ nguyên random seed.
