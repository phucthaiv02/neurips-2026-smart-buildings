# Bài toán và dữ liệu

Nguồn: [bài báo cuộc thi](https://arxiv.org/abs/2610.18511), [trang chính thức](https://smart-buildings-2026.github.io/competition/competition.html), [starter kit](https://github.com/smart-buildings-2026/starter-kit).

## Bài toán

- Nộp **một policy** (code), không nộp file dự đoán. Kaggle chạy simulator và gọi policy mỗi bước.
- Policy phải chạy trên tòa nhà và khí hậu chưa từng thấy, không được tune riêng.
- Đơn vị tổ chức: NSF AI Institute in Dynamic Systems, Harvard, Columbia, University of Washington, Kajima. Giải pháp thắng có thể được thử trên 3 tòa nhà thật.

## Dữ liệu

| | Train (công khai) | Đánh giá (ẩn) |
|---|---|---|
| Tòa nhà | 6 (`building_train_1/2/4/5/7/8`) | 6 tòa mới, cùng bộ sinh và khoảng tham số |
| Trạm thời tiết | fairbanks, denver, atlanta, phoenix, singapore | 5 trạm khác, không có trong train |
| Kịch bản | 30 (mỗi tòa × 1 đoạn mỗi trạm) | 30 |

- Mỗi kịch bản: 30 ngày, bước 5 phút, **8.640 bước**, mọi bước đều được chấm.
- Thời tiết: dữ liệu trạm đo hàng giờ năm 2023, mỗi đoạn một cửa sổ 30 ngày khác nhau. Không có thời tiết tổng hợp.
- Trạm `minneapolis` có sẵn làm đầu vào cho bộ sinh nhưng không kịch bản nào dùng → dùng làm validation.
- Bộ sinh tòa nhà thay đổi: số tầng (2–4), kích thước, hình dạng, bố trí phòng, mật độ người (1–2 người/zone), 6 thuộc tính vật liệu vỏ, số AHU (mỗi AHU ≤ 10 zone). Có thể sinh không giới hạn tòa nhà để train.

Cấu trúc starter kit:

```
smart_buildings/
  smart_control/            # simulator
  baselines/                # policy tham chiếu; transfer_policy.py là chỗ viết policy
  buildings/                # 6 tòa nhà train (YAML)
  weather/                  # các đoạn thời tiết (8 đoạn mỗi trạm)
  training_scenarios/       # 30 kịch bản (config.gin), ~101 MB
  run_floor_generator.py    # sinh tòa nhà mới
  run_scenario_generator.py # ghép tòa nhà × thời tiết
  evaluate.py               # chạy policy trên kịch bản
baseline_returns.json       # return của 4 baseline trên 30 kịch bản
score_leaderboard.py        # tính điểm leaderboard
```

## Hành động (hybrid, kích thước thay đổi theo tòa nhà)

Với K AHU và V zone:

| Phần | Shape | Nội dung |
|---|---|---|
| Rời rạc | K + 1 | Bật/tắt từng AHU và hệ nước nóng (HWS) |
| Liên tục | 3K + 2 + 2V | Mỗi AHU: nhiệt độ gió cấp, áp suất tĩnh, tỉ lệ tuần hoàn. HWS: nhiệt độ nước, chênh áp bơm. Mỗi zone: damper, van reheat |

- Train: K = 3–9, V = 28–85 → rời rạc 4–10 bit, liên tục 67–199 giá trị.
- Liên tục chuẩn hóa về [-1, 1]. **Action 0 là điểm giữa, không phải "tắt"** (damper 0 = mở một nửa).
- Action liên tục **xếp xen kẽ**. Luôn đọc chỉ số từ `building_meta_data` (`temp_indices`, `pressure_indices`, `recirculation_indices`, `water_indices`, `zone_action_indices`).
- Simulator ép sàn thông gió: AHU chạy giữ áp suất tối thiểu, tuần hoàn ≤ 70%, damper ≥ 10%.
- AHU chỉ làm lạnh; nhiệt sưởi đến từ reheat ở zone, cần có gió để mang nhiệt.

## Quan sát

Vector phẳng, dài theo tòa nhà (283–703 phần tử ở train):

- Mỗi zone: nhiệt độ, setpoint sưởi/làm lạnh, damper, reheat, setpoint lưu lượng.
- Mỗi AHU: nhiệt độ gió cấp, áp suất tĩnh, lưu lượng (cảm biến + setpoint), tuần hoàn, số yêu cầu làm lạnh, trạng thái quạt.
- HWS: nhiệt độ nước cấp.
- Số người trong tòa nhà, cờ chế độ tiện nghi (đang / sắp).
- Nhiệt độ ngoài trời. Lưu ý: cảm biến ngoài trời của AHU trả về **Kelvin thô**, không chuẩn hóa.
- Giờ trong ngày, ngày trong tuần (mã hóa vòng).

Lúc khởi tạo, policy nhận thêm metadata: chỉ số action, tên trường quan sát, ánh xạ zone → AHU, mặt bằng, lịch, hằng số chuẩn hóa.

## Reward và metric

Mỗi bước:

```
r_t = 0.6 · tiện_nghi_t − 0.2 · chi_phí_năng_lượng_t − 0.2 · carbon_t   (r_t ≤ 0)
```

- Phạt tiện nghi bằng 0 khi mọi zone có người nằm trong dải; tuyến tính tới 5 K ngoài dải, sau đó bậc hai, **không có trần**.
- Tính theo số người trong từng zone, không theo diện tích.
- Ban đêm dùng dải eco rộng hơn.

Điểm leaderboard:

```
MAE = (1/|S|) · Σ_s | mean_t r_t^(s) |      (thấp hơn = tốt hơn)
```

Trung bình đều trên 30 kịch bản, không gom theo tòa nhà hay khí hậu. Simulator tất định nên mỗi kịch bản chạy đúng 1 lần.

## Nộp bài

```python
from kaggle_evaluation import smart_buildings_inference_server

class MyPolicy(smart_buildings_inference_server.Policy):
    def __init__(self, building_meta_data, action_spec): ...
    def get_action(self, obs) -> dict:
        # {"continuous_action": float32[3K+2+2V], "discrete_action": int32[K+1]}
        ...

smart_buildings_inference_server.SmartBuildingsInferenceServer(MyPolicy).run()
```

- Notebook Kaggle, không internet.
- Mỗi rollout ≤ **10 × thời gian của baseline bang-bang**.
- Code bị quét tìm ID tòa nhà hoặc chữ ký cảm biến hard-code → loại.
- Public LB chấm trên một tập con của 30 kịch bản đánh giá; private chấm đủ 30. Ban tổ chức chạy lại top 7 để xác minh.
- Đội tối đa 4 người.

## Lịch và giải thưởng ($40k)

| Mốc | Ngày |
|---|---|
| Round One (poster NeurIPS) | 13/11/2026 |
| Workshop NeurIPS, Paris | 13/12/2026 |
| Hạn cuối | 15/01/2027 |

- Leaderboard chính $25.000: hạng 1 $5.750, hạng 2 $3.750, hạng 3 $3.000, hạng 4–8 $2.500.
- Bài báo Round One: hạng 1 $1.000, 2 $750, 3 $500, 4–14 $250.
- Bài báo cuối: hạng 1 $3.000, 2 $1.750, 3 $750, 4–12 $500.
- Nộp bài báo qua [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SB26_Challenge).
