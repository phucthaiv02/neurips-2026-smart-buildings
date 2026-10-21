# Cách chấm điểm khác gì các cuộc thi mô phỏng khác

Khác biệt lớn nhất: điểm là **thiệt hại tuyệt đối, không chuẩn hóa**, lấy trung bình trên 30 kịch bản. Vài kịch bản khí hậu khắc nghiệt quyết định gần hết thứ hạng.

## So sánh

| | Kaggle Simulation (Halite, Lux AI, Kore) | RL benchmark (Procgen, NetHack, MineRL) | Điều khiển năng lượng (CityLearn, L2RPN) | **Smart Buildings 2026** |
|---|---|---|---|---|
| Đối thủ | Agent đấu agent | Không | Không | **Không** |
| Kiểu điểm | Rating kỹ năng (μ, σ) | Reward trung bình | Thường chuẩn hóa theo baseline | **Thiệt hại tuyệt đối** |
| Ngẫu nhiên | Mỗi trận một seed | Trung bình nhiều seed | Tùy cuộc thi | **Tất định, 1 lần/kịch bản** |
| Số lần đánh giá | Hàng trăm trận | Hàng trăm–nghìn episode | Vài chục kịch bản | **30 kịch bản** |
| Sau hạn chót | Rating còn chạy 1–2 tuần | Chấm một lần | Chấm một lần | **Chấm một lần, chạy lại top 7** |

## Năm điểm khác biệt

1. **Không có đối thủ.** Điểm chỉ phụ thuộc policy của mình, tính chính xác được trên train. Không có "meta" để chạy theo.
2. **Trung bình thô → kịch bản khó chiếm phần lớn điểm.** Với bản v5 của notebook tham chiếu: riêng `building_train_1/fairbanks_1` chiếm ~19% tổng điểm; 6 kịch bản Fairbanks (20% số kịch bản) chiếm ~40%.

   | Khí hậu | Phần trong tổng điểm (v5, train) |
   |---|---|
   | Fairbanks | 39,9% |
   | Phoenix | 18,7% |
   | Denver | 16,1% |
   | Atlanta | 13,6% |
   | Singapore | 11,7% |

3. **Phạt tiện nghi không trần → rủi ro đuôi.** Một tòa nhà lạ làm policy "vỡ" có thể bằng cả chục kịch bản tốt. Độ bền quan trọng hơn mức tối ưu.
4. **Tất định nhưng hỗn loạn.** Không có seed, nhưng thay đổi cực nhỏ vẫn làm điểm lệch 1–6% vì một quyết định bật/tắt đổi chiều. Không giảm được bằng chạy lại; chỉ kiểm tra được bằng nhiều tòa nhà tự sinh.
5. **Public LB là tập con nhỏ.** Dễ đánh lừa. Ví dụ: bản v5 có public LB 0,052 trong khi trung bình 30 kịch bản train là ~0,038.

## Hệ quả

- Bảng điểm xếp theo **phần đóng góp vào tổng**, không chỉ điểm từng kịch bản.
- Hàm mục tiêu khi tune = đúng metric (trung bình thô), kèm ràng buộc không kịch bản nào tệ hơn hẳn baseline.
- Validation chính: tòa nhà tự sinh × 5 trạm + Minneapolis. Public LB chỉ là tín hiệu phụ.
- Mọi so sánh cần nhóm null; chỉ nhận thay đổi có |z| ≥ 2.
