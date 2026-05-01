## 1. Cấu trúc thư mục

Cấu trúc khuyến nghị:

```text
project_folder/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── sales.csv
│   └── sample_submission.csv
└── outputs/
    ├── submission.csv
    ├── submission_ensemble.csv
    ├── submission_recursive_yoy.csv
    ├── submission_baseline.csv
    ├── prediction_components.csv
    ├── feature_importance_anchor_revenue.csv
    ├── feature_importance_anchor_revenue.png
    ├── forecast_revenue.png
    ├── forecast_cogs.png
    ├── cv_results.csv
    ├── cv_summary.csv
    └── shap_summary_anchor_revenue.png
```

Trong đó:

- `main.py`: file code chính để huấn luyện mô hình và tạo file dự báo.
- `requirements.txt`: danh sách thư viện cần cài đặt.
- `data/`: thư mục chứa dữ liệu đầu vào.
- `outputs/`: thư mục chứa toàn bộ kết quả sinh ra sau khi chạy code.

---


## 2. Cài đặt môi trường

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

Kích hoạt môi trường ảo trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

File `requirements.txt` nên bao gồm:

```text
numpy
pandas
matplotlib
scikit-learn
lightgbm
shap
```

Ghi chú: `shap` chỉ cần nếu muốn tạo biểu đồ SHAP bằng tùy chọn `--run-shap`.

---

## 3. Cách chạy nhanh để tạo file nộp Kaggle

Chạy lệnh sau từ thư mục gốc của project:

```bash
python main.py --data-dir data --output-dir outputs
```

Sau khi chạy xong, file chính để nộp Kaggle là:

```text
outputs/submission.csv
```

Đây là file quan trọng nhất, gồm 3 cột:

```text
Date,Revenue,COGS
```

File này đã được code kiểm tra để đảm bảo:

- Đúng số dòng như `sample_submission.csv`.
- Đúng thứ tự ngày như `sample_submission.csv`.
- Không có giá trị thiếu.
- Không có giá trị âm ở `Revenue` và `COGS`.

---

## 4. Pipeline mô hình

Pipeline gồm các bước chính sau:

### 4.1. Đọc và kiểm tra dữ liệu

Hàm `load_and_validate()` thực hiện:

- Đọc `sales.csv` và `sample_submission.csv`.
- Chuyển cột `Date` sang kiểu ngày.
- Sắp xếp dữ liệu train theo thời gian.
- Giữ nguyên thứ tự gốc của `sample_submission.csv`.
- Kiểm tra tên cột, số dòng, ngày trùng lặp và tính liên tục theo ngày.

Mục tiêu của bước này là đảm bảo dữ liệu đầu vào hợp lệ trước khi huấn luyện mô hình.

### 4.2. Tạo đặc trưng thời gian

Hàm `make_calendar_features()` tạo các đặc trưng lịch từ cột `Date`, bao gồm:

- `time_idx`: số ngày tính từ ngày đầu tiên trong tập train.
- `year`, `month`, `quarter`, `weekofyear`.
- `dayofyear`, `dayofmonth`, `dayofweek`.
- Cờ cuối tuần: `is_weekend`.
- Cờ đầu tháng/cuối tháng: `is_month_start`, `is_month_end`.
- Đặc trưng chu kỳ theo năm bằng sin/cos.
- Đặc trưng chu kỳ theo tuần bằng sin/cos.

Các đặc trưng sin/cos giúp mô hình học được tính mùa vụ theo tuần và theo năm.

### 4.3. Baseline seasonal naive

Mô hình baseline dùng phương pháp seasonal naive với độ trễ 365 ngày. Với mỗi ngày cần dự báo, giá trị được lấy từ cùng thời điểm năm trước, sau đó điều chỉnh bằng hệ số tăng trưởng gần đây.

Baseline được dùng cho cả `Revenue` và `COGS`.

File tạo ra:

```text
outputs/submission_baseline.csv
```

### 4.4. Recursive LightGBM theo hướng YoY

Mô hình recursive LightGBM dự báo `Revenue` từng ngày trong tương lai. Sau khi dự báo một ngày, giá trị dự báo đó được thêm vào lịch sử để dùng cho các ngày tiếp theo.

Các đặc trưng chính:

- Lag dài hạn: `28`, `56`, `91`, `182`, `364`, `365`, `366`, `730` ngày.
- Rolling mean, std, min, max, median với các cửa sổ `28`, `56`, `91`, `182`, `365` ngày.
- Tỷ lệ giữa các lag như `lag365_to_lag730` và `lag364_to_lag365`.
- Đặc trưng lịch và mùa vụ.

Mặc định, mô hình học theo chế độ `yoy_ratio`, tức là học phần chênh lệch log giữa doanh thu hiện tại và doanh thu cùng kỳ năm trước. Cách này giúp mô hình tập trung vào tăng trưởng theo năm thay vì học trực tiếp giá trị tuyệt đối.

File tạo ra:

```text
outputs/submission_recursive_yoy.csv
```

### 4.5. Direct seasonal-anchor LightGBM

Đây là mô hình chính dùng để tạo file `submission.csv`.

Mô hình này không dự báo đệ quy từng ngày. Thay vào đó, mỗi ngày tương lai được gắn với các giá trị tham chiếu từ cùng kỳ các năm trước, ví dụ:

- Khoảng 364, 365, 366 ngày trước.
- Khoảng 729, 730, 731 ngày trước.
- Khoảng 1094, 1095, 1096 ngày trước.

Các giá trị quá khứ được điều chỉnh bằng hệ số tăng trưởng YoY. Sau đó LightGBM học phần residual trên thang log:

```text
log1p(Revenue thực tế) - log1p(anchor_revenue)
```

Cách tiếp cận này giúp mô hình tận dụng mạnh tính mùa vụ theo năm, đồng thời giảm rủi ro tích lũy lỗi so với recursive forecasting.

File chính tạo ra:

```text
outputs/submission.csv
```

### 4.6. Ensemble

Code cũng tạo thêm một dự báo ensemble từ ba thành phần:

- Recursive LightGBM.
- Direct seasonal-anchor LightGBM.
- Seasonal naive baseline.

Trọng số mặc định:

```text
recursive = 0.20
anchor = 0.45
baseline = 0.35
```

File tạo ra:

```text
outputs/submission_ensemble.csv
```

File này dùng để tham khảo hoặc so sánh, nhưng file chính được khuyến nghị trong code là:

```text
outputs/submission.csv
```

---

## 5. Cross-validation

Để đánh giá mô hình đúng theo chiều thời gian, code hỗ trợ rolling time-series cross-validation.

Chạy lệnh:

```bash
python main.py --data-dir data --output-dir outputs --run-cv
```

Các mốc train-end được dùng trong code:

```text
2019-06-30
2020-06-30
2021-06-30
```

Với mỗi fold:

1. Dữ liệu trước hoặc bằng `train_end` được dùng để huấn luyện.
2. Mô hình dự báo một khoảng thời gian có cùng độ dài với tập test.
3. Kết quả dự báo được so sánh với dữ liệu thực tế trong `sales.csv`.
4. Các chỉ số được tính gồm:
   - MAE
   - RMSE
   - R2

Các mô hình được đánh giá trong CV:

- `SeasonalNaive365Trend`
- `RecursiveLightGBM_yoy_ratio`
- `AnchorResidualLightGBM`
- `Ensemble`

Output của cross-validation:

```text
outputs/cv_results.csv
outputs/cv_summary.csv
```

Trong đó:

- `cv_results.csv`: lưu kết quả chi tiết từng fold.
- `cv_summary.csv`: lưu kết quả trung bình theo từng mô hình.

Cross-validation này giúp kiểm tra hiệu quả mô hình trên các giai đoạn tương lai giả lập, tránh đánh giá sai do leakage thời gian.

---

## 6. Feature importance

Sau khi huấn luyện mô hình chính, code xuất feature importance của mô hình `AnchorResidualLightGBM`.

Các file được tạo:

```text
outputs/feature_importance_anchor_revenue.csv
outputs/feature_importance_anchor_revenue.png
```

Ý nghĩa:

- File `.csv` chứa danh sách đặc trưng và mức độ quan trọng tương ứng.
- File `.png` hiển thị top 25 đặc trưng quan trọng nhất.

Feature importance giúp giải thích mô hình đã học dựa trên những yếu tố nào, ví dụ:

- Giá trị anchor từ cùng kỳ năm trước.
- Đặc trưng mùa vụ theo ngày trong năm.
- Đặc trưng tháng, tuần, ngày trong tuần.
- Hệ số tăng trưởng YoY.

Phần này có thể đưa vào báo cáo kỹ thuật để giải thích các yếu tố dẫn động doanh thu.

---

## 7. SHAP

Code hỗ trợ tạo SHAP summary plot nếu cài đặt thư viện `shap` và chạy với tùy chọn `--run-shap`.

Lệnh chạy:

```bash
python main.py --data-dir data --output-dir outputs --run-shap
```

Hoặc chạy cả cross-validation và SHAP:

```bash
python main.py --data-dir data --output-dir outputs --run-cv --run-shap
```

Output SHAP:

```text
outputs/shap_summary_anchor_revenue.png
```

SHAP giúp giải thích tác động của từng đặc trưng lên dự báo của mô hình. So với feature importance thông thường, SHAP có ưu điểm là thể hiện cả hướng tác động và mức độ tác động của đặc trưng.

Trong báo cáo kỹ thuật, biểu đồ SHAP có thể được dùng để giải thích:

- Đặc trưng nào ảnh hưởng mạnh nhất đến dự báo doanh thu.
- Các yếu tố mùa vụ có đóng vai trò quan trọng hay không.
- Mô hình có phụ thuộc nhiều vào anchor cùng kỳ năm trước hay không.

---

## 8. Các file output

Sau khi chạy đầy đủ, thư mục `outputs/` có thể gồm các file sau:

| File | Ý nghĩa |
|---|---|
| `submission.csv` | File chính để nộp Kaggle. |
| `submission_ensemble.csv` | File dự báo bằng ensemble. |
| `submission_recursive_yoy.csv` | File dự báo bằng Recursive LightGBM. |
| `submission_baseline.csv` | File dự báo baseline seasonal naive. |
| `prediction_components.csv` | So sánh các thành phần dự báo. |
| `cv_results.csv` | Kết quả cross-validation theo từng fold. |
| `cv_summary.csv` | Trung bình metric theo từng mô hình. |
| `feature_importance_anchor_revenue.csv` | Bảng feature importance. |
| `feature_importance_anchor_revenue.png` | Biểu đồ top feature importance. |
| `forecast_revenue.png` | Biểu đồ Revenue train vs forecast. |
| `forecast_cogs.png` | Biểu đồ COGS train vs forecast. |
| `shap_summary_anchor_revenue.png` | Biểu đồ SHAP, chỉ có nếu chạy `--run-shap`. |

File nộp lên Kaggle:

```text
outputs/submission.csv
```

---

## 9. Tham số dòng lệnh

Một số tham số quan trọng:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--data-dir` | `data` | Thư mục chứa dữ liệu đầu vào. |
| `--output-dir` | `outputs` | Thư mục lưu kết quả. |
| `--run-cv` | Không bật | Chạy rolling time-series cross-validation. |
| `--run-shap` | Không bật | Tạo SHAP summary plot nếu đã cài `shap`. |
| `--n-estimators` | `1400` | Số cây của LightGBM. |
| `--learning-rate` | `0.018` | Learning rate của LightGBM. |
| `--target-mode` | `yoy_ratio` | Cách biến đổi target cho recursive model. |
| `--recursive-weight` | `0.20` | Trọng số recursive trong ensemble. |
| `--anchor-weight` | `0.45` | Trọng số anchor trong ensemble. |
| `--baseline-weight` | `0.35` | Trọng số baseline trong ensemble. |

Ví dụ thay đổi thư mục output:

```bash
python main.py --data-dir data --output-dir my_outputs
```

Ví dụ thay đổi số cây LightGBM:

```bash
python main.py --data-dir data --output-dir outputs --n-estimators 1800
```

---

## 10. Tái lập kết quả

Code đã cố định random seed:

```python
SEED = 42
np.random.seed(SEED)
```

LightGBM cũng được truyền `random_state=SEED`. Điều này giúp kết quả ổn định hơn giữa các lần chạy.

Để tái lập kết quả, cần đảm bảo:

1. Dùng cùng phiên bản dữ liệu.
2. Dùng cùng file code.
3. Cài đặt đúng các thư viện trong `requirements.txt`.
4. Không thay đổi thứ tự dòng trong `sample_submission.csv`.
5. Chạy cùng câu lệnh và cùng tham số.

---


## 11. Tóm tắt phương pháp

Phương pháp chính là **Direct Seasonal-Anchor LightGBM**. Mô hình xây dựng giá trị anchor từ cùng kỳ các năm trước, điều chỉnh theo tăng trưởng YoY, sau đó học phần sai lệch còn lại bằng LightGBM trên thang log.

Cách tiếp cận này phù hợp với bài toán doanh thu hằng ngày vì doanh thu thường có:

- Tính mùa vụ theo tuần.
- Tính mùa vụ theo tháng/năm.
- Xu hướng tăng trưởng hoặc suy giảm theo thời gian.
- Các biến động ngắn hạn quanh cùng kỳ năm trước.

Kết quả cuối cùng được lưu tại:

```text
outputs/submission.csv
```
