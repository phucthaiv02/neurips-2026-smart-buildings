# Baseline và tốc độ simulator

## Điểm baseline (30 kịch bản train)

Tính từ `baseline_returns.json` của starter kit bằng [scripts/baseline_scores.py](../scripts/baseline_scores.py). Điểm = trung bình trên kịch bản của |return / 8640|. Thấp hơn = tốt hơn.

| Baseline | Điểm | Mô tả |
|---|---|---|
| bang_bang | 0,2252 | Bật/tắt AHU quanh dải tiện nghi |
| pid | 0,0674 | Điều khiển tỉ lệ, áp suất quạt theo nỗ lực điều khiển |
| fixed_setpoint | 0,0659 | Một dải tiện nghi suốt ngày đêm |
| **pid_water** | **0,0541** | `pid` + reheat điều biến + reset nước nóng theo nhu cầu |
| Chọn baseline tốt nhất từng kịch bản | 0,0473 | Không nộp được; cho thấy lợi ích của thích nghi |

- `pid_water` thắng 18/30 kịch bản, `fixed_setpoint` thắng 12/30.
- Khó nhất: phoenix, fairbanks (0,13–0,21). Dễ nhất: nhiều kịch bản atlanta, singapore (~0,02).
- `all_off` có sẵn nhưng không tính vào dải tham chiếu.

## Từng kịch bản (điểm, thấp hơn = tốt hơn)

| Kịch bản | bang_bang | fixed_setpoint | pid | pid_water |
|---|---|---|---|---|
| building_train_1/atlanta_2 | 0,176 | 0,026 | 0,026 | 0,024 |
| building_train_1/denver_7 | 0,077 | 0,031 | 0,031 | 0,030 |
| building_train_1/fairbanks_1 | 0,287 | 0,147 | 0,151 | 0,212 |
| building_train_1/phoenix_5 | 0,268 | 0,130 | 0,154 | 0,134 |
| building_train_1/singapore_8 | 0,249 | 0,027 | 0,031 | 0,028 |
| building_train_2/atlanta_3 | 0,254 | 0,020 | 0,020 | 0,020 |
| building_train_2/denver_8 | 0,206 | 0,095 | 0,068 | 0,040 |
| building_train_2/fairbanks_2 | 0,186 | 0,098 | 0,093 | 0,080 |
| building_train_2/phoenix_6 | 0,294 | 0,166 | 0,105 | 0,071 |
| building_train_2/singapore_1 | 0,276 | 0,023 | 0,044 | 0,026 |
| building_train_4/atlanta_5 | 0,250 | 0,027 | 0,029 | 0,028 |
| building_train_4/denver_2 | 0,151 | 0,060 | 0,062 | 0,055 |
| building_train_4/fairbanks_4 | 0,141 | 0,031 | 0,032 | 0,032 |
| building_train_4/phoenix_8 | 0,152 | 0,029 | 0,029 | 0,029 |
| building_train_4/singapore_3 | 0,248 | 0,028 | 0,045 | 0,032 |
| building_train_5/atlanta_6 | 0,294 | 0,020 | 0,038 | 0,027 |
| building_train_5/denver_3 | 0,185 | 0,116 | 0,073 | 0,044 |
| building_train_5/fairbanks_5 | 0,201 | 0,017 | 0,017 | 0,017 |
| building_train_5/phoenix_1 | 0,133 | 0,037 | 0,045 | 0,033 |
| building_train_5/singapore_4 | 0,303 | 0,024 | 0,186 | 0,109 |
| building_train_7/atlanta_8 | 0,204 | 0,064 | 0,075 | 0,048 |
| building_train_7/denver_5 | 0,245 | 0,139 | 0,043 | 0,036 |
| building_train_7/fairbanks_7 | 0,186 | 0,071 | 0,079 | 0,054 |
| building_train_7/phoenix_3 | 0,273 | 0,160 | 0,051 | 0,044 |
| building_train_7/singapore_6 | 0,268 | 0,026 | 0,044 | 0,031 |
| building_train_8/atlanta_1 | 0,213 | 0,041 | 0,063 | 0,032 |
| building_train_8/denver_6 | 0,241 | 0,020 | 0,020 | 0,020 |
| building_train_8/fairbanks_8 | 0,198 | 0,081 | 0,080 | 0,077 |
| building_train_8/phoenix_4 | 0,308 | 0,200 | 0,186 | 0,158 |
| building_train_8/singapore_7 | 0,288 | 0,024 | 0,100 | 0,054 |

## Tốc độ simulator

Đo bằng [scripts/time_sim.py](../scripts/time_sim.py) trên 1 nhân CPU, trạng thái ổn định:

| Tòa nhà | s/bước | 1 episode (8.640 bước) |
|---|---|---|
| building_train_1 (39 zone, 4 AHU) | 0,043 | ~6 phút |
| building_train_2 (85 zone, 9 AHU) | 0,098 | ~14 phút |

- Thêm ~5–7 s nạp môi trường.
- Trên CPU Kaggle, policy PI của notebook tham chiếu mất 744 s cho 1 episode building_train_1 (~0,086 s/bước).
- Mỗi nhân chỉ ~10–25 bước/giây → RL thuần 10 triệu bước ≈ 8 ngày-nhân.
