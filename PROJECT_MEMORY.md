# 项目记忆 — SFG-BiCross 合成数据集流水线

> 更新日期：2026-08-12（最新：Phase 2 checkpoint fix 验证完成，待用户重启 kernel 重跑 Phase D）

---

## 1. 项目概述

### 核心目标

为 `zherewy.com`（内容创作平台：Asset/Prompt/Commerce 三类内容）构建合成数据集流水线，用于训练 SFG-BiCross 推荐算法。运行模式为 **`paper_no_id`**（纯内容特征，无 user/item ID embedding）。最终目标是验证 SFG-BiCross 的多任务学习 + 交叉注意力机制能否从特征域的差异化贡献中受益。

### 数据流水线

```
Phase A (user/item latent 生成)        → 随机 SVD 主题向量 + skill TF-IDF + 人口统计特征
  ↓
Phase B (行为生成 + 交互 enrichment)    → 注入时序 + 协同过滤信号 + 行为标签
  ↓
Phase C (KuaiRand 7+1 文件格式)        → 标准 KuaiRand-Pure 格式输出
  ↓
adapt_for_sfg.py                        → UUID→int64 映射 + 列名适配
  ↓
Phase D (SFG-BiCross 训练)             → 用户手动执行
```

### 关键约束

| 约束 | 说明 |
|------|------|
| **不训练模型** | 用户明确："不用帮我训练，你先把所有的环境以及代码准备好，由我自己手动训练" |
| **先讨论再改代码** | 所有改动先分析方案、讨论优劣、确认方向后再实现 |
| **不偏离研究方向** | 始终对齐 SFG-BiCross `paper_no_id` 验证目标 |
| **KuaiRand 格式兼容** | 输出必须兼容 SFG-BiCross notebook 的读取逻辑 |

### 环境

| 用途 | Python 路径 |
|------|------------|
| 项目脚本（Phase B/C、adapt） | `D:\dev\projects\datasets_collection\.venv\Scripts\python.exe` |
| Phase D 训练（需要 PyTorch） | `D:\dev\projects\zhere_olap\.venv\Scripts\python.exe` |

### 关键文件

| 文件 | 用途 |
|------|------|
| `run_phase_b.py` | Phase B 独立脚本（Plan C 实现） |
| `run_phase_c.py` | Phase C 独立脚本（KuaiRand 格式输出） |
| `adapt_for_sfg.py` | UUID→int64 + 列名适配 + 缺失列补全 |
| `notebooks/phase_b_temporal_cf.ipynb` | Phase B notebook（已更新但用脚本执行） |
| `notebooks/phase_c_kuairand_format.ipynb` | Phase C notebook（long_view 已改为独立行为） |
| `notebooks/phase_d_train_sfg_bicross.ipynb` | SFG-BiCross 训练 notebook |
| `PHASE_D_README.md` | 训练指南 |
| `PROJECT_MEMORY.md` | 本文件 |

---

## 2. 数据质量演进史

### 阶段 0：初始数据集（问题严重）

**生成方式**：
- `compute_score` 使用 `sigmoid(raw)` → 分数被压缩到 [0.49, 0.67]，std=0.023
- `BEHAVIOR_FUNNEL` 在此基础上再对 score 做 sigmoid + Bernoulli 采样
- **双重 sigmoid 问题**：第一次 sigmoid 压缩动态范围，第二次 sigmoid 进一步压缩 → 行为标签近乎随机

**结果**：
- GBDT feature→engagement AUC：**0.516**（接近随机）
- Cohen's d < 0.17（所有行为几乎无区分度）
- 模型训练效果极差

### 阶段 1：K=8 放大修复

**改动**：`compute_score` 中 `sigmoid(raw)` → `sigmoid(8.0 * raw)`
- 分数动态范围扩大 ~4 倍，std 从 0.023 → 0.081
- BEHAVIOR_FUNNEL 阈值重新校准（a 和 b 参数大幅调整）

**结果**：
- GBDT feature→engagement AUC：0.618（仍不够）
- Cohen's d：0.25-0.75
- ms→engagement AUC：0.724

**暴露的深层问题**：所有行为共享同一个 `sigmoid(a * score - b)` 公式，不同行为之间的唯一区别是 `a` 和 `b` 参数不同。这在概率层面导致行为间相关性 >0.98，经过 Bernoulli 采样后相关性 ≈ 0。多任务学习没有共享信号可用，也没有差异化信号可学。

### 阶段 2：Plan C（当前实现）

详见第 3 节。核心改进：
- `long_view` 成为独立行为
- A_COMMON + 逐物品偏移替代 BEHAVIOR_FUNNEL
- 添加交互特征到 user/item latent

**结果**：ms→engagement AUC = **0.7338**（已达天花板）

### 阶段 3：方向 C（待实现）

详见第 4 节。核心改进：
- 不同行为依赖不同特征子集
- 70% 专属 + 30% 共享的混合架构

**预期**：feature→behavior GBDT AUC 显著提升，模型训练可学到更丰富的特征→行为映射

---

## 3. Plan C 实现细节（已实现）

### 3.1 架构设计

**核心思路**：所有行为共享同一个 `raw_score`（来自 `compute_raw`），每个行为加一个小的 item-dependent 偏移，再经各自的 sigmoid 阈值生成概率。

```
p_behavior = sigmoid( A_COMMON × (raw_score + shift_item[behavior]) - b_behavior )
```

**为什么选这个方案（当时讨论的三个选项）**：
- 选项 A（互动特征 + 行为差异化）：添加特征，但行为仍共享同一 score → 选了这个
- 选项 B（调整模型架构）：改动太大，偏离研究目的 → 不选
- 选项 C（改训练策略）：用户不需要我们改训练侧 → 不选

**为什么不偏离 KuaiRand-Pure 方向**：
- KuaiRand-Pure 的核心约束是"无 ID embedding，只有内容特征"
- 我们的改动只在行为标签层面（让标签更有区分度），特征仍然是纯内容特征
- 不引入任何 ID 信息或外部信号

### 3.2 关键函数

```python
def compute_raw(user_vec, item_idx):
    """返回原始分数（sigmoid 之前）"""
    t = cosine_similarity(user_theme, item_theme)   # 主题匹配
    p = 1.0 - abs(budget - item_price)               # 价格匹配
    s = cosine_similarity(user_skill, item_skill)    # 技能匹配
    q = quality_sensitivity × item_quality            # 品质匹配
    ct = dot(item_ct_onehot, user_ct_pref)            # 内容类型匹配
    raw = (w_theme×t + w_price×p + w_skill×s +
           w_quality×q + w_ct×ct +
           w_pop×item_popularity - w_fresh×(1-freshness))
    return raw

def compute_score(user_vec, item_idx):
    """match_score = sigmoid(8.0 × raw)，K=8 放大动态范围 ~4x"""
    return sigmoid(8.0 * compute_raw(user_vec, item_idx))
```

### 3.3 行为参数（经过多轮仿真调参）

```python
A_COMMON = 8.0   # 与 match_score 的 K 保持一致

# 逐物品偏移（零中心化，不影响整体行为率）
item_shift_like    = 0                                   # 参考行为
item_shift_long    = 0.12 × is_asset                     # Asset 内容更长的观看
item_shift_fav     = 0.08 × quality                      # 高质量 → 更多收藏
item_shift_comment = 0.15 × is_prompt × user_social      # Prompt + 社交倾向 → 评论
item_shift_share   = 0.08 × popularity + 0.05 × freshness # 热门+新鲜 → 分享
item_shift_buy     = 0.20 × is_commerce                  # 商业内容 → 购买
item_shift_view    = 0.02 × popularity                   # 热门 → 点击

# 阈值参数 (b) — 调参目标见下表
BEHAVIOR_SPEC = {
    'view':       (b=1.1,  shift=item_shift_view,    commerce_only=False),
    'like':       (b=3.4,  shift=item_shift_like,    commerce_only=False),
    'long_view':  (b=3.4,  shift=item_shift_long,    commerce_only=False),
    'fav':        (b=4.4,  shift=item_shift_fav,     commerce_only=False),
    'comment':    (b=5.4,  shift=item_shift_comment, commerce_only=False),
    'share':      (b=5.1,  shift=item_shift_share,   commerce_only=False),
    'buy':        (b=6.9,  shift=item_shift_buy,     commerce_only=True),
}
```

### 3.4 关键设计决策及原因

| 决策 | 原因 |
|------|------|
| `long_view` 成为独立行为 | 之前从 `fav\|(like & high_score)` 派生，与 like/fav 高度耦合。独立后可提供额外的行为维度，让多任务学习有更多差异化信号 |
| 所有偏移零中心化 | 确保偏移只改变哪些 item 获得行为，不改变整体行为率。行为率由 b 阈值单独控制 |
| comment 偏移 × `social_tendency` | 让用户社交倾向直接影响评论概率，创造用户级别的行为差异 |
| buy 仅限 commerce item | 非商业内容购买概率应为 0，避免模型学到虚假信号 |
| `A_COMMON = 8.0` 和 `K=8.0` 一致 | 保持 raw_score 的动态范围在行为生成和 match_score 中一致 |

### 3.5 调参过程

经过 **5+ 轮仿真迭代**才确定最终参数：

1. **第一轮**：沿用旧的 BEHAVIOR_FUNNEL 格式，comment/share/buy rate 严重偏离（8%/15.5%/17.4% vs 目标 4%/5%/1%）
2. **第二轮**：试了不同 A 值 per behavior → 行为间概率相关性 ≈ 0 → 完全独立，不符合现实
3. **第三轮**：同一 A（=8.0），不同 b → 相关性 0.03-0.08（只有 Bernoulli 噪声，没有信号层面的相关性）
4. **第四轮**：增大偏移量（0.12-0.20）→ 概率相关性 0.71-0.98，二元相关性 0.01-0.11 → **这个范围最合适**
5. **第五轮**：在真实数据上 5000 样本验证，确认所有指标达标

### 3.6 交互特征

行为生成后，从 `interactions_enriched` 聚合统计，添加到 `user_latent.csv` 和 `item_latent.csv`：

**用户特征**（9 个）：`user_n_interact`, `user_avg_ms`, `user_std_ms`, `user_like_rate`, `user_long_rate`, `user_fav_rate`, `user_comment_rate`, `user_share_rate`, `user_buy_rate`

**物品特征**（8 个）：`item_n_interact`, `item_avg_ms`, `item_like_rate`, `item_long_rate`, `item_fav_rate`, `item_comment_rate`, `item_share_rate`, `item_buy_rate`

**目的**：为模型提供用户活跃度和物品流行度的统计信号，辅助预测。

### 3.7 最终结果（479,038 条交互）

#### 行为率

| 行为 | 实际 | 目标 | 状态 |
|------|------|------|------|
| view | 68.0% | 68% | ✓ |
| like | 21.4% | 22% | ✓ |
| long_view | 23.3% | 22% | ✓ |
| fav | 9.8% | 10% | ✓ |
| comment | 3.8% | 4% | ✓ |
| share | 5.4% | 5% | ✓ |
| buy | 0.9% | 1% | ✓ |

#### 质量指标

| 指标 | 值 | 说明 |
|------|-----|------|
| ms→engagement AUC | 0.7338 | 自洽性检查通过 (>0.72) |
| ms→strong_action AUC | 0.7145 | |
| ms→like AUC | 0.6979 | |
| ms→long_view AUC | 0.7164 | |
| ms→fav AUC | 0.7031 | |
| ms→comment AUC | 0.6773 | 低 rate 行为天然 AUC 偏低 |
| ms→share AUC | 0.7151 | |
| ms→buy AUC | 0.7241 | commerce-only 约束提升区分度 |
| engagement rate | 46.5% | 合理 |
| engagement ≠ strong | 13.0% | long_view 独立贡献 |
| 行为间二元相关性 | 0.03-0.10 | 部分独立 ✓ |
| Jaccard | 0.067 (>5%: 79.5%) | 协同过滤信号充足 ✓ |

#### 文件输出

```
data/
├── interactions_enriched.csv       (479,038 rows, 10 cols)
├── user_latent.csv                 (6,006 users, 99 features)
├── item_latent.csv                 (39,000 items, 97 features)
├── user_clusters.csv
├── kuairand/                       (8 files, KuaiRand format)
│   ├── user_features_pure.csv
│   ├── video_features_basic_pure.csv
│   ├── log_standard_train_pure.csv
│   ├── log_standard_val_pure.csv
│   ├── log_standard_test_pure.csv
│   ├── log_random_test_pure.csv
│   ├── kuairand_video_captions.csv
│   └── kuairand_video_categories.csv
└── kuairand_adapted/               (10 files, int64 IDs)
    ├── user_id_mapping.csv
    ├── video_id_mapping.csv
    └── ... (训练就绪)
```

---

## 4. Feature→Behavior AUC 差距：方向 C 方案（待实现）

### 4.1 问题诊断

当前 `ms→engagement AUC = 0.73` 是 **自洽性检查**：match_score 预测自己生成的标签有多准。这不是模型训练时看到的信号质量。

真正关键的是 **feature→behavior AUC**：原始用户/物品特征（不经过 match_score 公式）通过 GBDT 预测行为的 AUC。这个值决定了模型能从数据中学到多少有效信号。

**根本原因**：

```
真实 KuaiRand: 用户真实偏好 → 多样化行为（特征捕捉真实偏好信号，多路径）
     vs
我们的合成数据: 随机 SVD 向量 → match_score(单一公式) → 行为标签（单一路径瓶颈）
```

所有 7 种行为都由同一个 `compute_raw` + 小偏移驱动。模型训练时，它能学到的上限就是 match_score 的信息量——而这已经被压缩成一个标量了。

**具体表现**：虽然我们有 90+ 维特征，但标签中蕴含的信号只有 match_score 那 1 维。模型需要先学会"这些高维特征 = match_score"这个隐含映射，才能进一步预测行为。这是一个不必要的弯路。

### 4.2 三个候选方案

在讨论中我们考虑了三个方向：

| 方向 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A | 特征直接参与行为生成（但仍通过同一 score） | 实现简单 | 仍是单瓶颈，AUC 提升有限 |
| B | 增加 match_score 的非线性（交互项） | 略微丰富信号 | 仍在压缩到一个标量，治标不治本 |
| **C** | **不同行为依赖不同特征子集** | **本质解决单瓶颈问题** | 实现稍复杂，需要仔细调参 |

### 4.3 推荐方案：方向 C（混合架构）

**核心公式**：

```
p_like    = sigmoid( 0.7 × f_like(theme_sim, freshness)        + 0.3 × logit(match_score) )
p_fav     = sigmoid( 0.7 × f_fav(quality, skill_sim)            + 0.3 × logit(match_score) )
p_comment = sigmoid( 0.7 × f_comment(social_tendency, is_prompt) + 0.3 × logit(match_score) )
p_share   = sigmoid( 0.7 × f_share(freshness, popularity)       + 0.3 × logit(match_score) )
p_buy     = sigmoid( 0.7 × f_buy(price_match, is_commerce)      + 0.3 × logit(match_score) )
p_long    = sigmoid( 0.7 × f_long(theme_sim, is_asset)          + 0.3 × logit(match_score) )
```

其中 `f_behavior()` 是每个行为专属的评分函数，使用 **2-4 个特定的特征**（不是全部特征），权重也是独立的。

**70/30 比例的考量**：
- 70% 专属信号 → 不同行为有真正独立的特征→行为通路，GBDT AUC 显著提升
- 30% 共享信号 → 行为间保持 0.1-0.3 的自然相关性，多任务学习的共享编码器仍有价值

### 4.4 为什么方向 C 最契合研究方向

1. **SFG-BiCross 的核心假设**：不同特征域（theme/skill/quality/CT）对不同行为的预测贡献不同。我们直接让这个假设在数据生成层面成立，模型必须学会利用交叉注意力来区分。

2. **不偏离 paper_no_id**：仍然不使用任何 ID embedding。行为的差异化来自不同特征子集的组合方式，而不是外部信号。

3. **多任务学习有价值**：70/30 混合确保行为不是完全独立（那会让多任务学习退化为独立模型），也不是完全相同（那会失去多任务的意义）。

### 4.5 为什么不选 A 或 B

- **方向 A**：特征作为"微调"附加在同一 score 上 → 瓶颈仍是那个 score，特征只是噪声级的修正。GBDT 看到的信号和现在差不多。
- **方向 B**：加交互项让 match_score 更非线性 → 但仍然是 1 维标量。信息没有增加，只是变换了形式。GBDT 本来就能学非线性关系，加交互项对它没帮助。

### 4.6 实施注意事项（讨论中确认）

- 专属函数 `f_behavior()` 的权重需要**单独调参**，不能让某种行为的 rate 偏离目标
- 需要在 logit 空间混合（`logit(match_score)`），而不是概率空间
- 每种行为的专属特征选择需要**有现实依据**（不能随机选），要反映真实场景中该行为的驱动因素
- 调参时先用仿真验证（5000-10000 样本），确认 rate 和 AUC 达标后再跑全量

### 4.7 状态

**仅讨论，尚未实现。** 等待用户确认后再编码。

---

## 5. 历史问题及解决记录

### 5.1 Cell 9：captions 文件列名不匹配

**问题**：Notebook 期望 `['final_video_id', 'caption', 'show_cover_text', 'duration']`，我们的文件是 `['video_id', 'caption_text', 'cover_text']`。

**解决**：在 `adapt_for_sfg.py` 第 6 节中重命名列 + 添加 `duration` 列（从日志中的 `duration_ms` 聚合）。

### 5.2 Cell 11：upload_dt NaT 错误

**问题**：`video_features_basic_pure.csv` 缺少 `upload_dt` 列 → `pd.to_datetime(None)` = NaT → `.dt.year` 报错 `AttributeError`。

**解决**：在 `adapt_for_sfg.py` 第 2 节中，从日志中提取每个视频的首次事件时间，减去随机 1-30 天作为 upload_dt。同时补全 `video_duration`, `server_width`, `server_height`, `upload_type`, `visible_status`, `music_type` 列。

### 5.3 Cell 13：is_lowactive_period 缺失

**问题**：Notebook 期望 29 个额外用户特征列，包括 `is_lowactive_period`, `is_live_streamer`, `is_video_author`, `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days`, `onehot_feat0..17`, 范围列等。

**解决**：在 `adapt_for_sfg.py` 第 3 节中：
- `is_lowactive_period` = 0（必须为 0 才能通过 active_mask）
- `is_live_streamer` = 5% 随机采样
- `is_video_author` = 10% 随机采样
- 社交数字特征 = lognormal 分布
- `user_active_degree` = 按交互次数 qcut 分为 5 档（`new/low_active/middle_active/high_active/full_active`）
- `onehot_feat0..17` = 随机 0/1

### 5.4 Cell 16：tag 缺失

**问题**：items dataframe 缺少 `tag` 列。

**解决**：在 `adapt_for_sfg.py` 第 4 节中添加 `vf['tag'] = ''`（空字符串，`parse_tags` 返回空列表）。

### 5.5 Cell 35：No random_train targets

**问题**：`PHASE2_SPLIT_DATE = "2022-05-01"`，但所有数据日期都是 2026 年 → 没有 random 数据被分配到训练集。

**解决**：改为 `"20260722"`（random log 日期范围 20260623-20260811 的中点）。结果：random_train=70,682 行，random_val=49,409 行。

### 5.6 adapt_for_sfg.py：rng_user.integers 不存在

**问题**：`numpy.random.RandomState` 没有 `integers` 方法（那是 `numpy.random.Generator` 的方法）。

**解决**：改为 `rng_user.randint(0, 2, ...)`。

### 5.7 UnicodeEncodeError（GBK）

**问题**：在 Windows 终端输出 Unicode 字符时报 GBK 编码错误。

**解决**：使用 `python -c "..."` 代替 heredoc，避免在代码中使用 Unicode 字符（用 ASCII 替代）。

### 5.8 Phase D Cell 37：best_model.pt 未找到（Phase 2 未保存 checkpoint）

**发现日期**：2026-08-12

**错误信息**：`FileNotFoundError: .../best_model.pt` at Cell 37 (P1 vs P2 evaluation)

**根因**：Phase 2 训练（Cell 36）在 `random_val` 上验证时，所有 metrics（AUC/NDCG/recall/hitrate）均为空，导致 `primary_value = NaN`，checkpoint 保存条件 `improved = np.isfinite(primary_value) and ...` 永远不为 True，`best_model.pt` 从未被保存。

深层原因：Phase C 的 `log_random_test_pure.csv` 中所有行为标签（`is_click`/`is_like`/`is_follow`/`is_comment`/`is_forward`/`long_view`）均为 0。Notebook 中 `engagement = max(long_view, is_like, is_follow, is_comment, is_forward)`，全部为 0 → engagement=0 → 无正例 → ranking metrics 无法计算。

**解决方案**：在 `run_phase_c.py` 中为随机日志添加隐式反馈（long_view）：

```python
# 基于物品质量和流行度计算隐式 long_view 概率
item_quality_raw = item_latent['quality_score'].values
item_pop_raw = item_latent['popularity'].values
ITEM_IMPLICIT_LOGIT = 5.0 * item_quality_raw + 3.0 * item_pop_raw - 6.2
ITEM_IMPLICIT_P_LONG = 1.0 / (1.0 + np.exp(-ITEM_IMPLICIT_LOGIT))

# 对每个随机曝光：
has_long = int(rng.random() < p_long)
play_time = rng.randint(90000, 150000) if has_long else rng.randint(0, 30000)
```

- 偏置 -6.2 经校准后 `p_long mean=0.182`，接近真实 KuaiRand 的 ~17%
- `play_time_ms`：long_view=1 时 90-150s，long_view=0 时 0-30s

**验证结果**（2026-08-12 适配后）：

| 日志文件 | rows | engagement | strong_action | long_view |
|---------|------|------------|---------------|-----------|
| log_standard_train | 334,177 | 0.464 | 0.335 | 0.232 |
| log_standard_val | 71,335 | 0.467 | 0.337 | 0.233 |
| log_standard_test | 73,526 | 0.466 | 0.335 | 0.233 |
| log_random | 120,014 | **0.181** | 0.000 | **0.181** |

**已完成**：
1. `run_phase_c.py` 已重跑 ✅（2026-08-12 19:53）
2. `adapt_for_sfg.py` 已重跑 ✅（2026-08-12 19:53）
3. 磁盘数据验证通过 ✅（2026-08-12 19:53）：

| 文件 | rows | engagement | pos/neg |
|------|------|------------|---------|
| log_standard_train | 334,177 | 0.464 | 155,125 / 179,052 |
| log_standard_val | 71,335 | 0.467 | 33,286 / 38,049 |
| log_random | 120,014 | **0.181** | **21,761 / 98,253** |

**待用户执行**（明天）：
4. 重启 Jupyter kernel，从 Cell 6 开始重跑（或 Run All）
5. Cell 11 输出应确认 `test rows=120,014 engagement_rate=0.1813`（而非 0.0000）
6. Cell 36 Phase 2 训练应能计算 ranking metrics 并保存 `best_model.pt`
7. Cell 37 P1 vs P2 对比应能正常加载 checkpoint

**关键提醒**：仅重跑 Cell 34-37 不够，因为 `test_log` 变量在 Cell 6+11 时已加载到内存。必须从 Cell 6 开始重跑，或者直接 Restart Kernel & Run All。

**Why:** 此修复让随机日志具备真实的隐式正反馈信号，使 Phase 2 去偏训练能够学习曝光偏差校正，同时像真实 KuaiRand 一样支持 ranking 指标评估。

**How to apply:** 修复已完成。用户重跑 notebook 即可验证。

---

## 6. 数据格式规范

### 6.1 KuaiRand 标准输出（Phase C）

| # | 文件 | 说明 |
|---|------|------|
| 1 | `user_features_pure.csv` | 用户纯内容特征（99 features + user_id） |
| 2 | `video_features_basic_pure.csv` | 物品纯内容特征（97 features + video_id + video_type） |
| 3 | `log_standard_train_pure.csv` | 反馈日志训练集（时间前 70%） |
| 4 | `log_standard_val_pure.csv` | 反馈日志验证集（中间 15%） |
| 5 | `log_standard_test_pure.csv` | 反馈日志测试集（后 15%） |
| 6 | `log_random_test_pure.csv` | 随机曝光日志（Phase 2 去偏，每用户 15-25 条） |
| 7 | `kuairand_video_captions.csv` | 物品文本描述（caption_text + cover_text） |
| 8 | `kuairand_video_categories.csv` | 二级分类层级 |

### 6.2 行为→KuaiRand 字段映射

| Phase B 行为 | KuaiRand 字段 | 说明 |
|-------------|--------------|------|
| view | is_click | 点击/查看 |
| like | is_like | 点赞 |
| fav | is_follow | 收藏→关注 |
| comment | is_comment | 评论 |
| share | is_forward | 分享→转发 |
| — | is_hate | 恒为 0（无负反馈） |
| long_view | long_view | **Plan C：独立行为，非派生** |

### 6.3 adapt_for_sfg.py 适配后的列名差异

adapt 脚本添加的列（notebook 期望但原始数据没有的）：

| 文件 | 添加的列 |
|------|---------|
| user_features | `is_lowactive_period`, `is_live_streamer`, `is_video_author`, `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days`, `*_range` (4), `onehot_feat0..17` (18), 重写 `user_active_degree` |
| video_features | `upload_dt`, `video_duration`, `server_width`, `server_height`, `upload_type`, `visible_status`, `music_type`, `tag` |
| captions | 重命名: `video_id→final_video_id`, `caption_text→caption`, `cover_text→show_cover_text`; 新增 `duration` |
| categories | 展开为 4 级分类 (id/name/prob), L3/L4 为空 |

### 6.4 Phase D 训练配置速查

| 参数 | 值 |
|------|-----|
| MODEL_MODE | `paper_no_id` |
| D_MODEL | 512 |
| NUM_HEADS | 8 |
| SELF layers | 2 |
| CROSS layers | 1 |
| FFN | 2048 |
| BATCH | 64 |
| MAX_HISTORY | 30 |
| EPOCHS | 15 |
| LR | 1.5e-4 |
| PHASE2_SPLIT_DATE | `20260722` |
| Primary Metric | NDCG@10 |

---

## 7. 讨论决策记录

### 7.1 关于数据质量优化方案的选择

**背景**：初始数据集 feature→engagement AUC = 0.516（接近随机），需要优化。

**讨论的三个方向**：
1. 添加互动特征 + 差异化行为标签
2. 调整模型架构（改 SFG-BiCross）
3. 改训练策略（early stopping、学习率调度等）

**用户问题**："这几件事是需要修改模型算法还是只对数据进行处理还是别的"
**结论**：只需要处理数据，不改模型。因为我们的目标是验证 SFG-BiCross，改模型就偏离了研究目的。

### 7.2 关于 KuaiRand-Pure 方向的偏离风险

**用户问题**："如果按这样调整数据集的话会不会偏离 KuaiRand-Pure 的数据集的方向"
**结论**：不会。KuaiRand-Pure 的核心约束是"无 ID embedding，只有内容特征"。我们在行为标签层面优化（让标签更有区分度），不引入 ID 信息或外部信号。

### 7.3 关于 0.72 AUC 天花板的讨论

**用户问题**："当前怎么达到天花板 0.72 呢，需要做什么改动呢（仅讨论，先不改代码）"
**讨论内容**：
- ms→engagement AUC = 0.72 是 match_score 预测自己生成的标签的能力上限
- 这是自洽性天花板，受限于 match_score 公式只使用了特征的一部分信息
- Plan C 实施后达到了 0.7338，略微超过了这个天花板

### 7.4 关于 Phase B→D 的运行流程

**用户问题**："我应该只运行 phaseD 的文件内容就行，还是从 B 到 D 都重新运行了一下"
**结论**：Plan C 改变了 Phase B 的行为生成逻辑和 Phase C 的 long_view 处理方式，所以需要 B → C → adapt → D 全部重跑。

### 7.5 关于方向 C 的选择

**用户问题**："你更推荐哪个，哪个不会偏离我们的方向（仅讨论）"
**推荐**：方向 C（70% 特征专属 + 30% 共享 match_score）
- 最贴近真实用户行为（不同行为由不同因素驱动）
- 完美契合 SFG-BiCross 的核心假设验证
- 不偏离 paper_no_id 方向
- 混合比例保证了多任务学习的价值

### 7.6 关于运行环境

**用户问题**："为什么当前的.py文件都在 zhere_olap 的虚拟环境运行，当前我们不是在 datasets_collection 中的项目中吗"
**发现**：`datasets_collection` 项目有自带的 `.venv`（安装了 pandas、numpy、scikit-learn、scipy），之前错误地使用了 `zhere_olap` 的 venv。后续所有脚本改为使用项目自己的 `.venv`。只有 Phase D 训练（需要 PyTorch）用 `zhere_olap\.venv`。

---

## 8. 用户偏好总结

1. **先讨论再写代码**：所有改动必须先分析问题、列出候选方案、讨论优劣、确认方向 → 再动手
2. **不训练模型**：只准备数据和代码，用户自己在 PyCharm 中手动执行 Phase D notebook
3. **研究方向优先**：任何改动都以"是否对齐 SFG-BiCross paper_no_id 验证目标"为第一评判标准
4. **不偏离 KuaiRand-Pure**：改动限于行为标签质量，不引入 ID embedding 或外部信号
5. **用项目自己的 .venv**：`datasets_collection\.venv` 运行脚本；`zhere_olap\.venv` 仅用于训练
6. **仿真验证后再全量跑**：调参先在 5000-10000 样本上仿真，确认指标达标再跑全量 479K 数据
