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
| [docs/06-rl.md](docs/06-rl.md) | Thiết kế RL: đặc trưng theo thực thể, residual PPO, BC/DAgger, cách chạy, thời gian |

## Notebooks

Mỗi notebook tự chứa toàn bộ code (ghi ra `sb_*.py` bằng `%%writefile`) và chạy độc lập: local cạnh bản clone starter kit, hoặc trên Kaggle với dữ liệu cuộc thi.

| File | Việc | Ra |
|---|---|---|
| [00_eval_reference_30.ipynb](notebooks/00_eval_reference_30.ipynb) | Policy tham chiếu (LB 0,01582) trên 30 kịch bản, nhóm null, đo thời gian | Điểm, tách thành phần, mức nhiễu |
| [01_candidates_screen.ipynb](notebooks/01_candidates_screen.ipynb) | Sàng mọi version so với tham chiếu, xác nhận 3 version tốt nhất trên 30 kịch bản | Điểm từng version |
| [02_param_search.ipynb](notebooks/02_param_search.ipynb) | CMA-ES 11 tham số, kiểm tra trên 24 kịch bản giữ lại | Điểm bộ tham số tốt nhất |
| [policies/*.ipynb](notebooks/policies) | Mỗi version một notebook: chấm 30 kịch bản; trên Kaggle chạy inference server để **Submit** | Điểm train + điểm LB |

Notebook được sinh từ `sbx/` bằng `python notebooks/build_notebooks.py`; sửa code trong `sbx/` rồi build lại.

## RL

[`sbx/rl/`](sbx/rl) · [docs/06-rl.md](docs/06-rl.md). Residual PPO trên controller 0,01582 (mạng khởi tạo = đúng controller), và BC/DAgger cho mạng chạy độc lập. Một mạng (83k tham số, ~1 ms/bước) dùng chung cho mọi zone/AHU nên chạy được mọi tòa nhà.

```bash
pip install -r requirements-rl.txt
python scripts/train_rl.py ppo --workers 4 --iters 200 --out runs/ppo_residual
python scripts/train_rl.py eval runs/ppo_residual/best.pt --workers 4
```

## Scripts

| File | Việc |
|---|---|
| [scripts/baseline_scores.py](scripts/baseline_scores.py) | Tính điểm leaderboard của các baseline từ `baseline_returns.json` |
| [scripts/time_sim.py](scripts/time_sim.py) | Đo thời gian mỗi bước của simulator |
| [scripts/train_rl.py](scripts/train_rl.py) | Huấn luyện PPO / BC, chấm checkpoint trên 30 kịch bản |
| [scripts/make_rl_scenarios.py](scripts/make_rl_scenarios.py) | Sinh thêm kịch bản (tòa nhà × đoạn thời tiết chưa dùng) cho RL |

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
