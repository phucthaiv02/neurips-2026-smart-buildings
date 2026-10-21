# NeurIPS 2026 Smart Buildings Challenge

Ghi chú nghiên cứu và kế hoạch cho cuộc thi [Smart Buildings Competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings) trên Kaggle (NeurIPS 2026 Competition Track).

Nhiệm vụ: viết **một** policy điều khiển HVAC chạy được trên các tòa nhà và khí hậu chưa từng thấy, cân bằng tiện nghi, năng lượng và carbon. Điểm là thiệt hại trung bình mỗi bước, càng gần 0 càng tốt.

## Tình trạng

| Mốc | Ngày |
|---|---|
| Round One (poster NeurIPS) | 13/11/2026 |
| Workshop Paris | 13/12/2026 |
| Hạn cuối | 15/01/2027 |

| Tham chiếu | Điểm |
|---|---|
| Baseline tốt nhất của ban tổ chức (`pid_water`, train) | 0,0541 |
| Notebook công khai tốt nhất (public LB) | 0,01708 |
| Mục tiêu | < 0,017 |

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/roadmap.html](docs/roadmap.html) | Lộ trình dạng trực quan: timeline, checklist, biểu đồ (mở bằng trình duyệt) |
| [docs/01-competition.md](docs/01-competition.md) | Bài toán, dữ liệu, không gian quan sát/hành động, reward, metric, lịch, giải thưởng |
| [docs/02-scoring.md](docs/02-scoring.md) | Cách chấm điểm khác gì các cuộc thi mô phỏng khác, và hệ quả chiến lược |
| [docs/03-baselines.md](docs/03-baselines.md) | Điểm baseline trên 30 kịch bản train, tốc độ simulator |
| [docs/04-reference-notebook.md](docs/04-reference-notebook.md) | Phân tích notebook 0,01708 (PI theo zone, giữ AHU, bug economizer) |
| [docs/05-plan.md](docs/05-plan.md) | Kế hoạch 4 giai đoạn: việc cần làm, cổng quyết định, ngân sách tính toán |

## Scripts

| File | Việc |
|---|---|
| [scripts/baseline_scores.py](scripts/baseline_scores.py) | Tính điểm leaderboard của các baseline từ `baseline_returns.json` |
| [scripts/time_sim.py](scripts/time_sim.py) | Đo thời gian mỗi bước của simulator |

## Cài môi trường

Starter kit cần Python 3.12 (numba 0.60 không chạy trên 3.13).

```bash
git clone https://github.com/smart-buildings-2026/starter-kit.git
cd starter-kit
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -e .
python -m smart_buildings.evaluate --policies pid_water --buildings building_train_1 --weathers atlanta_2
```

Config `.gin` dùng đường dẫn tương đối, nên mọi script phải chạy từ thư mục gốc của starter kit.

## Nguồn

- [The Smart Buildings Challenge (arXiv 2610.18511)](https://arxiv.org/abs/2610.18511)
- [Trang chính thức](https://smart-buildings-2026.github.io/competition/)
- [Starter kit](https://github.com/smart-buildings-2026/starter-kit)
- [Notebook "Smart Buildings Probe DS K07"](https://www.kaggle.com/code/malyshevdanil/smart-buildings-probe-ds-k07)
