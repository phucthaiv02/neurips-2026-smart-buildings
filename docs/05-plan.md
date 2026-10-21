# Kế hoạch

Bản trực quan: [roadmap.html](roadmap.html).

Hướng: tune controller viết tay trước → imitation learning → RL sau cùng. RL thuần từ đầu quá đắt (simulator 10–25 bước/giây/nhân) và khó vượt mức 0,017 đã có.

```
21/10 ─ GĐ0 ─ 24/10 ──── GĐ1 ──── 5/11 ── GĐ2 ── 13/11 (Round One) ───────── GĐ3 ───────── 10/01 ─ chốt ─ 15/01 (hạn cuối)
```

## Giai đoạn 0 · Nền đo lường (21–23/10)

- [ ] Chạy notebook 0,017 trên 30 kịch bản train
- [ ] Tách thiệt hại còn lại: tiện nghi / năng lượng / carbon
- [ ] Bộ chấm chạy song song + nhóm null (vài lần chạy gần như giống hệt)
- [ ] Xếp kịch bản theo phần đóng góp vào tổng điểm

**Ra:** bảng điểm từng kịch bản + mức nhiễu.

## Giai đoạn 1 · Tune tham số (24/10–4/11)

- [ ] Gom ~10 tham số của controller thành một vector
- [ ] Sinh thêm tòa nhà bằng bộ sinh; giữ Minneapolis để kiểm tra
- [ ] CMA-ES tìm **một bộ chung** cho mọi tòa nhà (nộp được ngay)
- [ ] Tune **riêng từng kịch bản** = "thầy" cho giai đoạn 2
- [ ] Lọc nhanh bằng episode 7 ngày, chấm lại ứng viên tốt bằng 30 ngày

**Cổng:** so điểm tune chung với tune riêng.
- Chênh lớn (vd. 0,017 so với 0,012) → chọn tham số theo tòa nhà đáng giá → làm giai đoạn 2.
- Chênh nhỏ → tham số không phải nút thắt → sửa logic controller hoặc sang giai đoạn 3.

## Giai đoạn 2 · Imitation + nộp Round One (5–13/11)

- [ ] Đặc trưng tòa nhà quan sát được lúc chạy: số zone, số AHU, diện tích, nhiệt độ ngoài vài giờ đầu, tốc độ nóng/nguội
- [ ] Mô hình đoán bộ tham số tốt từ đặc trưng (học từ các bộ tối ưu của giai đoạn 1)
- [ ] Kiểm tra trên tòa nhà + thời tiết chưa thấy
- [ ] Kiểm tra thời gian chạy ≤ 10 × bang-bang, nộp Kaggle
- [ ] Viết bài poster

**Ra:** bài nộp + poster trước 13/11.

## Giai đoạn 3 · Policy nơ-ron (14/11–10/01)

- [ ] Mạng chia sẻ tham số theo zone và AHU (chạy được mọi kích thước tòa nhà)
- [ ] Behavior cloning → DAgger từ controller tốt nhất
- [ ] Residual RL / fine-tune nhắm phần năng lượng + carbon còn lại
- [ ] Tuỳ chọn: thầy "nhìn trước tương lai" (cần sao chép được trạng thái simulator)
- [ ] Distill cho vừa giới hạn thời gian

**Cổng:** nếu không vượt thầy → giữ bài giai đoạn 2.

## Chuỗi thầy → trò

```
Controller có tham số ─→ Tune riêng từng kịch bản ─→ Mô hình đoán tham số ─→ Policy nơ-ron (BC/DAgger) ─→ Residual RL
       (GĐ1)                (thầy, GĐ1)                 (trò, GĐ2)                  (GĐ3)                  (GĐ3)
```

Học trò khó giỏi hơn thầy, nên thầy phải dùng thông tin policy nộp bài không có (đáp án tham số cho từng kịch bản, hoặc biết trước tương lai).

## Ngân sách tính toán

| Việc | Ước tính |
|---|---|
| 1 episode 30 ngày | 6–14 phút-nhân |
| Tune riêng 30 kịch bản × 100 lần | ≈ 400 giờ-nhân (≈ 100 nếu lọc bằng 7 ngày) |
| RL 10 triệu bước | ≈ 190 giờ-nhân cho 1 lần chạy |

## Câu hỏi mở

- Có bao nhiêu nhân CPU / GPU để chạy?
- Simulator có sao chép trạng thái được không?
- Policy PI (744 s/episode trên Kaggle) có lọt giới hạn 10 × bang-bang không?
