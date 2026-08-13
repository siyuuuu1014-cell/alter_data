# 合成数据集生成方案 — zherewy.com → SFG-BiCross

> v2 修订版 | 2026-08-11

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [数据概况](#2-数据概况)
3. [核心方案：Latent Preference Model](#3-核心方案latent-preference-model)
4. [实施路线](#4-实施路线)
5. [目标算法：SFG-BiCross](#5-目标算法sfg-bicross)
6. [最终输出](#6-最终输出)
7. [待确认决策](#7-待确认决策)

---

## 1. 背景与问题

### 1.1 为什么需要合成数据？

当前 `zhere_olap/data/recommendation/` 下的交互数据存在**致命缺陷**：

| 问题 | 具体表现 | 影响 |
|---|---|---|
| 标签与特征无统计关联 | 用户的 like/fav/share/buy 行为是**随机分配**给任意 item 的 | 任何 ML 模型看到的都是噪声，无法超过随机推荐 |
| 无协同过滤信号 | 用户之间几乎没有共同交互过的 item | 协同过滤、双塔模型全部失效 |
| 缺少时间戳 | behavior 表有交互但无 event_time_ms | 无法构建序列模型 |
| 缺少随机曝光日志 | 全部为反馈日志，无随机曝光对照 | SFG-BiCross Phase 2 去偏训练无法进行 |

**举例**：用户 A 喜欢"赛博朋克风格的高精度 3D 资产"，但当前数据中 A 的 like 行为被随机分配给了一个低质量手绘卡通素材。模型无论多复杂，学到的都是噪声。

### 1.2 解决方案思路

不随机分配标签，而是构造一条**因果链**：

```
用户真实特征 → 潜在偏好向量 → 与物品的匹配分数 → 行为概率 → 标签
```

---

## 2. 数据概况

### 2.1 数据源

所有数据从 `D:\dev\projects\zhere_olap\data\recommendation\` 读取。

### 2.2 用户数据：hybrid_user_profile.csv

| 指标 | 数值 |
|---|---|
| 总用户数 | 6,006 |
| active_old | 5,858 (97.5%) — 历史丰富、偏好稳定的老用户 |
| high_potential_new | 148 (2.5%) — 算法识别的高潜新用户 |

**关键字段**（除基础画像外）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `purchase` | JSON 数组 | 已购买的 item ID 列表 |
| `assets` | JSON 数组 | 已持有的 asset ID 列表 |
| `works` | JSON 数组 | 作品 ID 列表 |
| `prompt` | JSON 数组 | 已持有的 prompt ID 列表 |
| `commerce` | JSON 数组 | 已参与的 commerce ID 列表 |
| `rate` | float (2.0-4.2) | 用户综合评分 |
| `latitude` / `longitude` | float | 精确地理位置 |
| `is_active_old` | 0/1 | 是否活跃老用户 |
| `is_high_potential_new` | 0/1 | 是否高潜新用户 |
| `user_segment` | string | 用户分段标签 |

> **关键发现**：`purchase`、`assets`、`works`、`prompt`、`commerce` 这 5 个字段包含**真实的 item ID 数组**，可以通过 `content_features` 和 `published_*` 表回查 item 属性，从而**推导**（而非推断）用户的真实偏好。

### 2.3 内容数据（3 张表，共 39,001 条）

| 表名 | 行数 | 主要内容 |
|---|---|---|
| `published_asset` | 15,001 | 数字资产（title, description, theme, size, definition, duration, pricing, 互动计数） |
| `published_prompt` | 12,001 | AI Prompt（title, description, theme, rate, pricing, participants, 互动计数） |
| `published_commerce` | 12,001 | 商业需求（organization, activity, place, budget, context, jds, requirements, range, 互动计数） |

**辅助表**：`content_features` (39,001 行) — target_id, content_type, tags, purchase_rate, like_rate, fav_rate, comment_rate, share_rate

### 2.4 交互数据（3 张表，共 60,003 条）

| 表名 | 行数 | 字段 |
|---|---|---|
| `u2a_behavior` | 25,001 | user_id, asset_id, view, like, fav, share, comment, rate, buy, sell |
| `u2p_behavior` | 20,001 | 同上 |
| `u2c_behavior` | 15,001 | 同上 |
| `u2u_behavior` | 15,001 | user_id × 2, follow, message |

**人均交互**：60,003 ÷ 6,006 ≈ **10 条/人**（v1 为 3.75 条/人）

### 2.5 v1 → v2 关键变化

| 维度 | v1 | v2 |
|---|---|---|
| 用户表 | user_profile_features (16,004) | hybrid_user_profile (6,006) |
| 用户类型 | 含大量冷用户 | 全部为活跃/高潜用户 |
| Latent 来源 | 从文本推断 | 从真实 ID 数组**推导** |
| 行为基线 | 低（混合冷用户） | 上调（全部活跃） |
| 验证方式 | LR AUC > 0.65 | 增加 hold-out 真实 ID 验证 |

---

## 3. 核心方案：Latent Preference Model

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     数据生成 Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Step 1          Step 2           Step 3         Step 4     │
│  ┌─────────┐    ┌───────────┐    ┌─────────┐    ┌───────┐  │
│  │ User    │    │ Item      │    │ Match   │    │行为   │  │
│  │ Latent  │───▶│ Latent    │───▶│ Score   │───▶│标签   │  │
│  │ Factors │    │ Factors   │    │ 加权内积 │    │漏斗生成│  │
│  │ (8个)   │    │ (8个)     │    │         │    │       │  │
│  └─────────┘    └───────────┘    └─────────┘    └───────┘  │
│       ↑              ↑                          │          │
│  真实 ID 数组    content_features                ↓          │
│  回查推导        文本 embedding             验证: hold-out  │
│                                               + LR AUC     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Step 1: User Latent Factors（8 个因子，~140 维）

| # | Factor | 维度 | 含义 | 来源（active_old） | 来源（high_potential_new） |
|---|---|---|---|---|---|
| 1 | `content_type_pref` | 3 | 对 asset/prompt/commerce 的偏好权重 | 用户持有 ID 的类型分布 | 从 intro + designation 推断 |
| 2 | `theme_affinity` | ~50 | 对各主题的亲和度向量 | 回查 item 的 theme 字段，统计分布 | 从 intro 文本语义映射 |
| 3 | `quality_sensitivity` | 1 | 对内容质量的敏感度 | 用户 rate vs 持有 item quality 的相关性 | level + experience 推断 |
| 4 | `budget_level` | 1 | 消费预算等级 | level + fans/follow + purchase 频率 | 同左 |
| 5 | `location_preference` | 1 | 是否偏好同城内容 | 用户 location vs 持有 item location 的重合率 | 同左 |
| 6 | `skill_match_vector` | ~32 | 用户技能向量 | goodat 文本 → embedding | 同左 |
| 7 | `activity_level` | 1 | 整体活跃度 | fans/follow/liked/fav/comment 归一化 | 同左 |
| 8 | `social_tendency` | 1 | 社交倾向 | comment_count / shared_count 比例 | 同左 |

> **active_old 的优势**：Factor 1、2、3、5 可以从真实 ID 数组**精确计算**，而非从文本推断。

### 3.3 Step 2: Item Latent Factors（8 个因子，~140 维）

| # | Factor | 维度 | 来源 |
|---|---|---|---|
| 1 | `content_type` | 1 (cat) | content_features.content_type |
| 2 | `theme_embedding` | ~50 | title + description + theme → text embedding |
| 3 | `quality_score` | 1 | purchase_rate + like/fav/comment/share_rate 加权 |
| 4 | `price_tier` | 1 | pricing 字段分桶（免费/低价/中价/高价） |
| 5 | `location` | 1 (cat) | 各表的 location 字段 |
| 6 | `skill_requirements` | ~32 | description + requirements + jds → text embedding |
| 7 | `freshness` | 1 | publish 时间（需模拟或从时间戳推断） |
| 8 | `popularity` | 1 | view_count + like_count 综合 |

### 3.4 Step 3: Match Score 计算

```
match_score(u, i) = σ(
    w₁ · cos_sim(u.theme_affinity, i.theme_embedding)        // 主题匹配
  + w₂ · (1 - |u.budget_level - i.price_tier|)                // 价格匹配
  + w₃ · cos_sim(u.skill_match, i.skill_requirements)         // 技能匹配
  + w₄ · u.quality_sensitivity × i.quality_score              // 质量敏感度 × 物品质量
  + w₅ · u.content_type_pref[type(i)]                         // 内容类型偏好
  + w₆ · location_match(u, i)                                  // 同城加成 (0/1)
  - w₇ · i.freshness                                           // 新鲜度惩罚
  + w₈ · i.popularity                                          // 热门加成
)
```

- σ = sigmoid，输出范围 (0, 1)
- w₁~w₈ 是可调权重，通过真实 ID 数组拟合

### 3.5 Step 4: 漏斗式行为生成

match_score 越高，用户越可能产生深度互动：

```
行为      概率函数                        说明
──────    ──────────────────────────────   ──────────────────
view      σ(3 × score - 0.5)              门槛最低，几乎都会触发
like      σ(5 × score - 2.0)              需要一定匹配度
fav       σ(6 × score - 3.5)              更强偏好
comment   σ(5 × score - 3.5) × social     受社交倾向调节
share     σ(7 × score - 4.5)              最高门槛，极强偏好才触发
buy       σ(8 × score - 6.0)              仅 commerce 类型
```

**示例**：score = 0.7 时，P(view) ≈ 88%, P(like) ≈ 62%, P(fav) ≈ 35%, P(comment) ≈ 35% × social, P(share) ≈ 12%

**v2 调整**：active_old 用户阈值上调 ~0.3-0.5（原为 σ(3×score - 1)），反映高活跃特征。high_potential_new 用户保持较低阈值，模拟探索期行为。

### 3.6 协同信号保证

1. **热门 item 幂律分配**：少数热门 item 分配给大量用户（模拟头部效应）
2. **同 cluster 用户共享**：按 theme_affinity 聚类，同簇用户共享更多 item
3. **真实锚点继承**：从用户真实 ID 数组中的 item 共现关系扩充
4. **目标**：任意两用户的 Jaccard overlap > 5%

### 3.7 两段式策略

| 维度 | active_old (5,858 人) | high_potential_new (148 人) |
|---|---|---|
| Latent 来源 | 真实 ID 数组 → 数据驱动 | 人口统计 + 文本 → 推断 |
| 行为漏斗阈值 | 上调 | 保持低阈值 |
| 每用户交互数 | ~8-12 条 | ~20-30 条（需更多数据建立偏好画像） |
| 协同信号 | 自然形成 | 通过同 cluster 老用户桥接 |

> high_potential_new 仅 148 人，策略为：用 active_old 的 latent→behavior 模式训练模型，再将 high_potential_new 映射到同一 latent 空间。

### 3.8 验证策略

| 策略 | 方法 | 指标 |
|---|---|---|
| **策略 A（强验证）** | 从用户 ID 数组中 hold out 20%，用剩下 80% 推导 latent，检查 hold-out item 的预测 rank | Precision@20, NDCG@10 |
| **策略 B（基础验证）** | 用 latent → 生成 interaction labels → LR/XGBoost 恢复 latent 权重 | AUC > 0.65 |

> 策略 A 是真正的"预测用户未来行为"测试，比策略 B 更有说服力。

---

## 4. 实施路线

### Phase A-0: 数据探索（预计 1 个 session）

```
输入: hybrid_user_profile.csv, content_features.csv, published_*.csv
├── 解析 6,006 用户的 purchase/assets/works/prompt/commerce ID 数组
├── JOIN content_features 回查每个 item 的 content_type, tags, 各项 rate
├── JOIN published_* 回查 theme, pricing, location 等
├── 统计每用户: content_type 分布 | theme 分布 | quality 分布 | 价格档分布
└── 输出: 6,006 × N 的真实偏好画像矩阵
```

### Phase A-1: 构建 User Latent

```
输入: Phase A-0 偏好画像
├── active_old (5,858): 从真实 ID 分布计算 8 个 latent factor
├── high_potential_new (148): 从 intro/goodat/experience 文本 embedding 推断
└── 输出: 6,006 × ~140 的 user_latent 矩阵
```

### Phase A-2: 构建 Item Latent

```
输入: content_features + published_*
├── 提取 8 个 item factor
└── 输出: 39,001 × ~140 的 item_latent 矩阵
```

### Phase A-3: Match Score + 行为生成 + 验证

```
输入: user_latent, item_latent
├── 基于真实 ID 数组拟合权重 w₁~w₈
├── 计算所有 (u,i) pair 的 match_score
├── 按行为漏斗生成交互标签
├── 验证策略 A: hold-out 验证
├── 验证策略 B: LR/XGBoost AUC
└── 达标: AUC > 0.65 → 进入 Phase B
```

### Phase B: 注入时序和协同信号

```
├── 生成时间戳（模拟 3-6 个月的时间序列）
├── 热门 item 幂律分布 + 同簇共享
├── 用户 session 切分（每用户 3-5 个 session）
└── 验证: CF precision@20 > 0.10
```

### Phase C: 输出 SFG-BiCross 7 文件

```
├── 格式化 user_features_pure.csv
├── 格式化 video_features_basic_pure.csv
├── 格式化 log_standard_train_pure.csv / log_standard_val_pure.csv
├── 格式化 log_random_test_pure.csv
├── 格式化 kuairand_video_captions.csv
├── 格式化 kuairand_video_categories.csv
└── 字段名和逻辑对齐 KuaiRand 格式
```

### Phase D: 迭代调优

```
├── 在 SFG-BiCross 上试跑
├── 调整 latent factor 权重
├── 调整行为漏斗阈值
└── 验证模型收敛（loss 下降 + NDCG 提升）
```

---

## 5. 目标算法：SFG-BiCross

### 5.1 算法概述

SFG-BiCross (Semantic Field-Gated Bidirectional Cross-Attention) 是 KuaiRand 数据集上使用的纯内容推荐模型。

### 5.2 模型流程

```
1. FieldTokenEncoder   → 用户/物品多字段 → d_model 维 token
2. FieldGate           → 每个字段的软特征选择 gate ∈ (0, 2)
3. Self-Attention ×2   → QK-Norm + 可学习温度
4. Bidirectional Cross → U2I + I2U 同时更新
5. HybridMaskedPooling → [mean, max, attention] → d_model
6. Highway Gate        → 融合 pre-cross 和 post-cross 向量
7. CandidateAwareDIN   → 候选物品与历史事件交互注意力
8. DCN-V2 + Deep Tower → 特征交叉 + 深层网络
9. 4 Prediction Heads  → engagement / strong_action / long_view / click
```

### 5.3 关键配置

| 参数 | 值 |
|---|---|
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
| Early Stopping | patience=3, metric=NDCG@10 |

### 5.4 损失函数

```
Total = 1.0 × BCE(engagement)
      + 0.4 × PairwiseLogistic
      + 0.2 × BCE(strong_action)
      + 0.15 × BCE(long_view)
      + 0.05 × BCE(click)
```

### 5.5 两阶段去偏

- **Phase 1**：反馈日志训练全部参数
- **Phase 2**：冻结底层 encoder，在随机曝光日志上微调上层
- P1 vs P2 在 random_val 上公平比较

### 5.6 两种模式

| 模式 | 说明 |
|---|---|
| `paper_no_id` | 纯内容特征，无 ID embedding — **我们的目标模式** |
| `hybrid_max_accuracy` | 启用 User/Item ID embedding (128维) |

---

## 6. 最终输出

### 6.1 7 个文件

| # | 文件名 | 内容 |
|---|---|---|
| 1 | `user_features_pure.csv` | 用户纯内容特征（无 ID） |
| 2 | `video_features_basic_pure.csv` | 物品纯内容特征 |
| 3 | `log_standard_train_pure.csv` | 反馈日志训练集 |
| 4 | `log_standard_val_pure.csv` | 反馈日志验证集 |
| 5 | `log_random_test_pure.csv` | 随机曝光日志测试集（用于 Phase 2 去偏） |
| 6 | `kuairand_video_captions.csv` | 物品文本描述 |
| 7 | `kuairand_video_categories.csv` | 多级分类层级 |

### 6.2 日志必需字段

```
user_id, video_id, date, time_ms, is_click, long_view, is_like,
is_follow, is_comment, is_forward, is_hate, play_time_ms, duration_ms
```

### 6.3 文本字段映射

| 目标字段 | 来源 |
|---|---|
| `caption_text` | title + " " + description |
| `cover_text` | requirements（commerce）或 theme + location（asset/prompt） |
| `category_path_text` | theme → 需扩充为 2-3 级分类 |

---

## 7. 待确认决策

| # | 问题 | 当前方案 | 状态 |
|---|---|---|---|
| 1 | Latent factor 维度 | 8 个因子 ~140 维 | ⬜ 待确认 |
| 2 | 协同信号强度 | 3-8% user-user Jaccard overlap | ⬜ 待确认 |
| 3 | 行为漏斗基准概率 | active_old: σ(3×score - 0.5), new: σ(3×score - 1) | ⬜ 待确认 |
| 4 | Phase A-0 启动 | 从 ID 数组解析开始 | ⬜ 待确认 |
| 5 | 执行位置 | `notebooks/` 下的 notebook？ | ⬜ 待确认 |

---

## 附录：数据 Schema 速查

### hybrid_user_profile.csv
```
user_id, is_active_old, is_high_potential_new, user_segment, hybrid_recommend,
segment_updated_at, name, birthday, level, gender, intro, latitude, longitude,
location, designation, experience, goodat, purchase, assets, works, prompt,
commerce, rate, register_at, fans_count, follow_count, liked_count, fav_count,
comment_count, shared_count, updated_at
```

### content_features.csv
```
target_id, content_type, tags, purchase_rate, like_rate, fav_rate,
comment_rate, share_rate, updated_at
```

### u2a_behavior.csv / u2p_behavior.csv / u2c_behavior.csv
```
user_id, {asset|prompt|commerce}_id, view, like, fav, share, comment,
rate, buy, sell
```
