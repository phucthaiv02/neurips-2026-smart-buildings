# Notebook tham chiếu: 0,01708

[Smart Buildings Probe DS K07](https://www.kaggle.com/code/malyshevdanil/smart-buildings-probe-ds-k07) (Malyshev Danil, Apache 2.0). Public LB **0,01708**. Không dùng RL: controller viết tay dựa trên `PIDWaterResetPolicy` của ban tổ chức, sửa từng lỗi một.

## Tiến trình

| Phiên bản | Thay đổi | Public LB |
|---|---|---|
| v3 | PID gốc, dải tiện nghi đọc từ cảm biến, giải mã nhiệt độ ngoài trời | 0,06439 |
| v4 | Giữ AHU làm lạnh tối thiểu 6 bước | 0,05882 |
| v5 | Giữ 288 bước (24 h) + trần làm lạnh ban đêm | 0,05195 |
| mới nhất | v5 + PI riêng cho từng zone | **0,01708** |

## Phát hiện chính

1. **Zone dao động mạnh.** Vòng zone của baseline chỉ có khâu P, mở damper hết cỡ khi lệch 2 K → nhiệt độ dao động 2–4 K mỗi 5–15 phút, zone ngoài dải 57–61% giờ làm. Tiện nghi chiếm 60% thiệt hại.
   - Sửa: PI riêng mỗi zone nhắm tâm dải (Kp 0,07, Ki 0,0035 mỗi bước), chống tích phân bão hòa, chạy từ 2 h trước giờ có người đến hết giờ làm. Một tín hiệu u: damper theo |u|, reheat theo −u. AHU được ép chạy trong giờ này.
   - Trên 5 kịch bản đầy đủ: 0,0204 so với 0,0483 của bản chỉ giữ AHU.
2. **AHU bật/tắt liên tục** (50–107 lần/ngày). Sửa: AHU bật để làm lạnh thì giữ ≥ 288 bước. Chỉ áp dụng khi làm lạnh; giữ cả lúc sưởi làm Fairbanks tệ đi.
3. **Bug economizer của ban tổ chức.** Cảm biến ngoài trời của AHU trả về Kelvin thô nhưng code giải mã như đã chuẩn hóa → economizer "thấy" ~3000 K, không bao giờ mở. Ai copy `pid_water` đều dính.
4. **Tinh chỉnh năng lượng:**
   - Trần làm lạnh ban đêm = trần ban ngày (tránh nóng dồn qua đêm).
   - Hạ setpoint sưởi thêm 2 K ngoài giờ làm.
   - Nhiệt độ gió cấp AHU theo nhu cầu PI trung bình các zone (cần sưởi → không làm lạnh gió).
   - Nhiệt độ nước nóng tăng theo nhu cầu reheat lớn nhất.
5. **Đánh giá nhiễu.** Thay đổi một phần trăm pascal có thể làm hai lần chạy lệch nhau 10 K sau vài ngày. Nhóm null dao động 1–6%/kịch bản. Chỉ nhận thay đổi có |z| ≥ 2.

## Thử nghiệm bị loại

- Reset áp suất quạt giữ lưu lượng: trong vùng nhiễu.
- Gió cấp lạnh hơn: Singapore tệ hơn 74%.
- Trễ bật AHU: tiện nghi tệ hơn (−6% đến −49%).

## Tham số bản nộp

```python
class SubmissionPolicy(TunedDemandSatPolicy, smart_buildings_inference_server.Policy):
    KP = 0.07
    KI = 0.0035
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    FORCED_PRESSURE = -1.0
    NIGHT_HEAT_DROP_K = 2.0
    PRESTART_H = 2
```

Đây là điểm xuất phát cho giai đoạn 1 của [kế hoạch](05-plan.md): các hằng số trên chính là vector tham số để tune.
