---
name: sfg-bicross-algorithm
description: KuaiRand_Pure_Final_paper_no_id_backup.ipynb 中 SFG-BiCross 推荐算法的完整架构分析
metadata:
  type: reference
  modified: 2026-08-10
---

# SFG-BiCross 算法架构

## 来源
`notebooks/Attention_test/KuaiRand_Pure_Final_paper_no_id_backup.ipynb`（在 zhere_olap 项目中）

## 核心架构：SFG-BiCross (Semantic Field-Gated Bidirectional Cross-Attention)

### 模型流程
1. FieldTokenEncoder: 用户/物品多字段 → d_model 维 token
   - Categorical → Embedding
   - Numeric → MLP(1→64→d_model) + LayerNorm
   - Tag → Embedding
   - Text → Linear(768→d_model) + GELU + LayerNorm
2. FieldGate: 每个字段的软特征选择 gate ∈ (0, 2)
3. Self-Attention (2 layers): QK-Norm + 可学习温度
4. Bidirectional Cross-Attention (1 layer): U2I + I2U 同时更新
5. HybridMaskedPooling: [mean, max, attention] → Linear → d_model
6. Highway Gate: 融合 pre-cross 和 post-cross 向量
7. CandidateAwareDIN: 候选物品与历史事件交互注意力
8. Match Layer + DCN-V2 (rank=64) + Deep Tower (512→256)
9. 4 个 Prediction Heads: engagement / strong_action / long_view / click

### 两种模式
- `paper_no_id`: 纯内容特征，无 ID embedding
- `hybrid_max_accuracy`: 启用 User/Item ID embedding (128维)

### 损失函数
Total = 1.0×BCE(engagement) + 0.4×PairwiseLogistic + 0.2×BCE(strong_action) + 0.15×BCE(long_view) + 0.05×BCE(click)

### 训练配置
- D_MODEL=512, NUM_HEADS=8, SELF=2层, CROSS=1层, FFN=2048
- BATCH=64, MAX_HISTORY=30, EPOCHS=15, LR=1.5e-4
- AdamW + Cosine Warmup + EarlyStopping(patience=3, metric=NDCG@10)

### 两阶段去偏 (Phase 2)
- Phase 1: 反馈日志训练
- Phase 2: 冻结底层 encoder，在随机曝光日志上微调上层
- P1 vs P2 在 random_val 上公平比较

### 数据要求（7 个文件）
- user_features_pure.csv, video_features_basic_pure.csv
- log_standard_*_pure.csv (train/val), log_random_*_pure.csv (test)
- kuairand_video_captions.csv, kuairand_video_categories.csv

### 日志必需字段
user_id, video_id, date, time_ms, is_click, long_view, is_like, is_follow, is_comment, is_forward, is_hate, play_time_ms, duration_ms

**Why:** 这是当前项目需要适配的目标算法，理解其 Schema 要求是合成数据设计的依据。

**How to apply:** 合成数据时必须输出这 7 个文件格式，所有字段名和逻辑保持一致。
