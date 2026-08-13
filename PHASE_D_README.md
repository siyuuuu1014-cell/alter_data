# Phase D: SFG-BiCross 训练指南

## 前置条件

- **Python 环境**: `.venv` (项目自带虚拟环境) 或 `D:\dev\projects\zhere_olap\.venv` (Python 3.13.9, PyTorch 2.13+cu130)
- **GPU**: NVIDIA GeForce RTX 5060 Laptop (8GB VRAM), CUDA 13.0
- **数据**: `data/kuairand_adapted/` (已适配为 int64 ID + YYYYMMDD 日期格式)

## 文件说明

```
datasets_collection/
├── adapt_for_sfg.py                          # 数据适配脚本（已完成）
├── data/kuairand_adapted/                    # 适配后的训练数据
│   ├── user_features_pure.csv                # 用户特征 (6,006 × 91)
│   ├── video_features_basic_pure.csv         # 物品特征 (39,000 × 91)
│   ├── log_standard_4_08_to_4_21_pure.csv    # 训练日志 (334,184 行)
│   ├── log_standard_4_22_to_5_08_pure.csv    # 验证日志 (71,363 行)
│   ├── log_random_4_22_to_5_08_pure.csv      # 随机曝光测试日志 (119,837 行)
│   ├── kuairand_video_captions.csv           # 物品文本
│   ├── kuairand_video_categories.csv         # 物品分类
│   ├── user_id_mapping.csv                   # UUID→int 映射表
│   └── video_id_mapping.csv                  # UUID→int 映射表
├── notebooks/
│   └── phase_d_train_sfg_bicross.ipynb       # 训练 notebook（已适配路径）
├── outputs_sfg_bicross/                      # 训练输出目录（自动创建）
└── text_cache/                               # MacBERT 文本缓存（首次运行下载）
```

## 运行步骤

### 1. 确认环境

在 PyCharm 中打开 `phase_d_train_sfg_bicross.ipynb`，确保 kernel 选择 `zhere_olap` 环境。

或在终端验证：
```bash
"D:\dev\projects\zhere_olap\.venv\Scripts\python.exe" -c "
import torch, transformers
print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')
print(f'Transformers {transformers.__version__}')
"
```

### 2. 首次运行：Debug 模式验证

Notebook 中 Cell 3 的配置已设为：
```python
DEBUG_MODE: bool = False        # ← 改为 True
DEBUG_USER_COUNT: int = 1200    # 使用 1200 用户
DEBUG_TRAIN_MAX_EVENTS: int = 100_000
DEBUG_EPOCHS: int = 3
RUN_SMALL_OVERFIT_CHECK: bool = False  # ← 改为 True
```

**建议首次运行**：
1. 修改 Cell 3 中 `DEBUG_MODE = True`
2. 修改 `RUN_SMALL_OVERFIT_CHECK = True`
3. 从头执行所有 cell
4. 确认 loss 收敛 + overfit check 通过
5. 预期耗时：~15 分钟

### 3. 全量训练

Debug 验证通过后：
1. 修改 `DEBUG_MODE = False`
2. 修改 `RUN_SMALL_OVERFIT_CHECK = False`
3. 重新执行所有 cell
4. 预期耗时：2-4 小时（15 epochs, batch=64）

### 4. 关键指标

训练过程中关注的指标：
- **NDCG@10**: 主指标，应该持续上升
- **Val Loss**: 应该持续下降
- **Engagement AUC**: > 0.70 为合格
- **strong_action AUC**: > 0.75 为合格

Early stopping: patience=3, 若 NDCG@10 连续 3 个 epoch 不提升则自动停止。

### 5. 输出文件

训练完成后 `outputs_sfg_bicross/` 下会生成：
- `model_best.pt` — 最佳模型权重
- `preprocessor_final_v1/` — 特征预处理器
- TensorBoard 日志

## 数据列名映射

适配后的 captions/categories 文件已对齐 notebook 期望：

| 文件 | 原始列 | 适配后列 |
|---|---|---|
| kuairand_video_captions | video_id, caption_text, cover_text | final_video_id, caption, show_cover_text, duration |
| kuairand_video_categories | video_id, category_level_1, category_level_2 | final_video_id + 4级分层 category (id/name/prob) |

- `duration`: 从日志中提取每个视频的 median duration_ms，prompts/commerce 默认为 60000ms
- L1 分类: Asset / Prompt / Commerce (3 类)
- L2 分类: 主题名称 (29 类)
- L3/L4: 空（我们的数据只有 2 层分类），prob=0.0

## 数据映射说明

如需将预测结果映射回原始 UUID：
```python
import pandas as pd
uid_map = pd.read_csv('data/kuairand_adapted/user_id_mapping.csv')
vid_map = pd.read_csv('data/kuairand_adapted/video_id_mapping.csv')
# int → UUID
original_user_id = uid_map[uid_map['user_id_int'] == pred_user_id]['user_id_orig'].iloc[0]
```

## 训练配置速查

| 参数 | 值 |
|---|---|
| MODEL_MODE | paper_no_id (纯内容) |
| D_MODEL | 512 |
| NUM_HEADS | 8 |
| SELF layers | 2 |
| CROSS layers | 1 |
| FFN | 2048 |
| BATCH | 64 |
| MAX_HISTORY | 30 |
| EPOCHS | 15 |
| LR | 1.5e-4 |
| Optimizer | AdamW + Cosine Warmup |
| Primary Metric | NDCG@10 |
