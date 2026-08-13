---
name: synthetic-dataset-design
description: 为 zherewy.com 平台设计适配 SFG-BiCross 算法的合成数据集
metadata:
  type: project
  modified: 2026-08-11
---

# 合成数据集设计 — zherewy.com → SFG-BiCross（v2 修订版）

## 数据源
从 `D:\dev\projects\zhere_olap\data\recommendation\` 读取以下文件：
- `hybrid_user_profile.csv` (6,006 用户: 5,858 active_old + 148 high_potential_new)
- `content_features.csv` (39,001 内容)
- `published_asset.csv` (15,001), `published_prompt.csv` (12,001), `published_commerce.csv` (12,001)
- `u2a_behavior.csv` (25,001), `u2p_behavior.csv` (20,001), `u2c_behavior.csv` (15,001)

## 核心问题
当前交互标签与用户/物品特征之间不存在统计上的因果关系，需要基于 Latent Preference Model 生成合成数据。

## 解决方案：基于潜在偏好模型（Latent Preference Model）的因果数据生成

### 核心思路
构建 "用户偏好 × 物品属性 → 交互概率" 的因果链，而非随机分配标签。

### 关键变更（v1 → v2）
1. **用户表替换**: user_profile_features.csv (16k) → hybrid_user_profile.csv (6k active/high-potential only)
2. **Latent Factor 提取升级**: 用户的 purchase/assets/works/prompt/commerce 列包含真实 item ID 数组，可回查 content_features 推导真实偏好，从"推断"变为"推导"
3. **两段式策略**: active_old (5,858) 从真实 ID 推导；high_potential_new (148) 从文本推断 + 老用户桥接
4. **行为漏斗阈值上调**: 活跃用户基线从 σ(3×score - 1) 上调至 σ(3×score - 0.5)
5. **验证升级**: 增加 hold-out 真实 ID 验证 + 原 LR AUC > 0.65

### User Latent Factors（8 个）
- content_type_pref (3维): 从 ID 类型分布计算
- theme_affinity (~50维): 回查 item theme 统计真实分布
- quality_sensitivity (1维): 用户 rate vs item quality 相关性
- budget_level (1维): 从 level + fans/follow 推断
- location_preference (1维): 是否偏好同城
- skill_match_vector (~32维): 用户 goodat → embedding
- activity_level (1维): 从社交计数归一化
- social_tendency (1维): comment/share 比例

### Item Latent Factors（8 个）
- content_type, theme_embedding, quality_score, price_tier, location, skill_requirements, freshness, popularity

### 行为漏斗
P(view) = σ(3×score - 0.5), P(like) = σ(5×score - 2.0), P(fav) = σ(6×score - 3.5), P(comment) = σ(5×score - 3.5) × social_tendency, P(share) = σ(7×score - 4.5)

### 最终输出
SFG-BiCross 兼容的 7 个文件: user_features_pure.csv, video_features_basic_pure.csv, log_standard_*_pure.csv, log_random_*_pure.csv, kuairand_video_captions.csv, kuairand_video_categories.csv

### 实施阶段
- Phase A-0: 数据探索（解析 ID 数组，JOIN 回查属性）
- Phase A-1: 构建 User Latent（数据驱动）
- Phase A-2: 构建 Item Latent
- Phase A-3: Match Score + 行为生成 + 验证
- Phase B: 注入时序和协同信号
- Phase C: 输出 SFG-BiCross 7 文件
- Phase D: 迭代调优

**Why:** 当前每张表的交互标签是随机分配的，与用户/物品特征不存在统计因果关系，任何 ML 模型都学不到有效信号。

**How to apply:** 从 Phase A-0 开始，解析 hybrid_user_profile 中的 ID 数组，回查 content_features 建立偏好画像。
