"""
Phase B standalone runner — Plan C version.
Generates enriched interactions with partially independent behavior labels
using the A_COMMON + per-item shifts approach.
"""
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
SEED = 42
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

# ============================================================
# Load data
# ============================================================
print('=== Loading data ===')
user_latent = pd.read_csv(f'{DATA_DIR}/user_latent.csv', index_col='user_id')
item_latent = pd.read_csv(f'{DATA_DIR}/item_latent.csv', index_col='item_id')
interactions = pd.read_csv(f'{DATA_DIR}/interactions_synthetic.csv')
with open(f'{DATA_DIR}/match_weights.json') as f:
    BEST_WEIGHTS = json.load(f)

# Fix double prefix
rename_map = {}
for c in user_latent.columns:
    if c.startswith('theme_theme_'):
        rename_map[c] = c.replace('theme_theme_', 'theme_')
user_latent.rename(columns=rename_map, inplace=True)

# Drop any stale interaction feature columns
for col in list(user_latent.columns):
    if col.startswith('user_') and (col.endswith('_rate') or col.endswith('_ms') or col.endswith('_interact')):
        del user_latent[col]
for col in list(item_latent.columns):
    if col.startswith('item_') and (col.endswith('_rate') or col.endswith('_ms') or col.endswith('_interact')):
        del item_latent[col]

print(f'Users: {len(user_latent):,}, Items: {len(item_latent):,}, Interactions: {len(interactions):,}')
iu = interactions.groupby('user_id').size()
print(f'Interactions per user (original): mean={iu.mean():.1f}, median={iu.median():.0f}')

# ============================================================
# 1. Power-law item popularity
# ============================================================
n_items = len(item_latent)
alpha = 1.3
ranks = np.arange(1, n_items + 1)
item_weights = ranks ** (-alpha)
item_probs = item_weights / item_weights.sum()
top5_n = int(n_items * 0.05)
top5_share = item_probs[:top5_n].sum()
print(f'Top 5% items ({top5_n:,}): {top5_share*100:.1f}% weight')

# ============================================================
# 2. User theme clustering
# ============================================================
THEME_COLS = [c for c in user_latent.columns if c.startswith('theme_')]
user_theme_vecs = user_latent[THEME_COLS].values
user_theme_norm = normalize(user_theme_vecs, norm='l2')

N_CLUSTERS = 25
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
user_clusters = kmeans.fit_predict(user_theme_norm)
user_latent['cluster'] = user_clusters
print(f'Clusters: {N_CLUSTERS}, sizes: {np.bincount(user_clusters).min()}-{np.bincount(user_clusters).max()}')

CLUSTER_SHARE_SIZE = 250
item_ids_arr = item_latent.index.values
cluster_item_pool = {}
for c in range(N_CLUSTERS):
    cluster_item_pool[c] = set(rng.choice(item_ids_arr, size=CLUSTER_SHARE_SIZE, replace=False))

# ============================================================
# 3. Timestamp injection
# ============================================================
TIME_START = np.datetime64('2026-03-01')
TIME_END = np.datetime64('2026-08-11')
TOTAL_DAYS = (TIME_END - TIME_START).astype(int)
print(f'Time range: {TIME_START} ~ {TIME_END} ({TOTAL_DAYS} days)')

HOUR_PROBS = np.ones(24)
HOUR_PROBS[18:24] = 3.0; HOUR_PROBS[8:12] = 1.5; HOUR_PROBS[12:18] = 2.0
HOUR_PROBS = HOUR_PROBS / HOUR_PROBS.sum()

user_sessions = {}
for uid in user_latent.index:
    n_sessions = rng.randint(3, 6)
    session_days = np.sort(rng.choice(TOTAL_DAYS, size=n_sessions, replace=False))
    events_per_session = rng.randint(15, 26, size=n_sessions)
    sessions = []
    for day, n_evt in zip(session_days, events_per_session):
        hour = rng.choice(24, p=HOUR_PROBS)
        minute = rng.randint(0, 60)
        base = TIME_START + np.timedelta64(int(day), 'D') + np.timedelta64(hour, 'h') + np.timedelta64(minute, 'm')
        sessions.append((base, n_evt))
    user_sessions[uid] = sessions

n_sess = [len(v) for v in user_sessions.values()]
n_evt = [sum(s[1] for s in v) for v in user_sessions.values()]
print(f'Sessions/user: mean={np.mean(n_sess):.1f}, Events/user: {np.mean(n_evt):.1f}, total={sum(n_evt):,}')

# ============================================================
# 4. Pre-compute score components + shifts (Plan C)
# ============================================================
SKILL_COLS = [c for c in item_latent.columns if c.startswith('skill_')]
CT_COLS = ['ct_published_asset', 'ct_published_prompt', 'ct_published_commerce']

item_theme_vecs = item_latent[THEME_COLS].values
item_skill_vecs = item_latent[SKILL_COLS].values
item_ct_onehot = item_latent[CT_COLS].values
item_quality = item_latent['quality_score'].values
item_price = item_latent['price_tier'].values
item_freshness = item_latent['freshness'].values
item_popularity = item_latent['popularity'].values
item_idx_map = {iid: i for i, iid in enumerate(item_ids_arr)}
item_is_commerce = item_latent['ct_published_commerce'].values.astype(bool)

def compute_raw(user_vec, item_idx):
    u_theme = user_vec[THEME_COLS].values.astype(float)
    u_skill = user_vec[SKILL_COLS].values.astype(float)
    u_ct = user_vec[CT_COLS].values.astype(float)
    qsens = float(user_vec['quality_sensitivity'])
    budget = float(user_vec['budget_level'])
    it, isk = item_theme_vecs[item_idx], item_skill_vecs[item_idx]
    ut_norm = np.linalg.norm(u_theme)
    t = 0.0 if ut_norm < 1e-12 else float(np.dot(it/(np.linalg.norm(it)+1e-12), u_theme/ut_norm))
    p = 1.0 - abs(budget - item_price[item_idx])
    us_norm = np.linalg.norm(u_skill)
    s = 0.0 if us_norm < 1e-12 else float(np.dot(isk/(np.linalg.norm(isk)+1e-12), u_skill/us_norm))
    q = qsens * item_quality[item_idx]
    ct = float(np.dot(item_ct_onehot[item_idx], u_ct))
    raw = (BEST_WEIGHTS['theme']*t + BEST_WEIGHTS['price']*p + BEST_WEIGHTS['skill']*s +
           BEST_WEIGHTS['quality']*q + BEST_WEIGHTS['ct']*ct +
           BEST_WEIGHTS['pop']*item_popularity[item_idx] -
           BEST_WEIGHTS['fresh']*(1.0-item_freshness[item_idx]))
    return float(raw)

# Plan C: Per-item behavior shifts
A_COMMON = 8.0
item_shift_like    = np.zeros(n_items)
item_shift_long    = 0.12 * item_ct_onehot[:, 0].astype(float)       # assets → longer viewing
item_shift_fav     = 0.08 * item_quality                             # high quality → more collections
item_shift_comment = 0.15 * item_ct_onehot[:, 1].astype(float)       # prompts (× social in loop)
item_shift_share   = 0.08 * item_popularity + 0.05 * item_freshness  # popular/fresh → more shares
item_shift_buy     = 0.20 * item_ct_onehot[:, 2].astype(float)       # commerce → purchases
item_shift_view    = 0.02 * item_popularity                          # popular → more clicks

for s in [item_shift_like, item_shift_long, item_shift_fav,
          item_shift_comment, item_shift_share, item_shift_buy, item_shift_view]:
    s -= s.mean()

BEHAVIOR_SPEC = {
    'view':       (1.1,  item_shift_view,    False),
    'like':       (3.4,  item_shift_like,    False),
    'long_view':  (3.4,  item_shift_long,    False),
    'fav':        (4.4,  item_shift_fav,     False),
    'comment':    (5.4,  item_shift_comment, False),
    'share':      (5.1,  item_shift_share,   False),
    'buy':        (6.9,  item_shift_buy,     True),
}

print(f'A_COMMON={A_COMMON}, behaviors: {list(BEHAVIOR_SPEC.keys())}')
for name, (b, s, _) in BEHAVIOR_SPEC.items():
    print(f'  {name:12s}: b={b:.1f}, shift std={s.std():.4f}, range=[{s.min():+.3f}, {s.max():+.3f}]')

# ============================================================
# 5. Generate enriched interactions (Plan C)
# ============================================================
HOT_FRAC, CLUSTER_FRAC, GLOBAL_FRAC = 0.08, 0.25, 0.40
GLOBAL_POOL_SIZE = 80
global_pool = rng.choice(item_ids_arr, size=GLOBAL_POOL_SIZE, replace=False)

print('\nGenerating enriched interactions (Plan C: A_COMMON + per-item shifts)...')
records = []
for i, uid in enumerate(user_latent.index):
    if uid not in user_sessions: continue
    user_vec = user_latent.loc[uid]
    uc = user_latent.loc[uid, 'cluster']
    social = float(user_vec['social_tendency'])
    user_orig = interactions[interactions['user_id'] == uid]['item_id'].values

    for session_base, n_events in user_sessions[uid]:
        session_items = []
        n_global = int(n_events * GLOBAL_FRAC)
        session_items.extend(rng.choice(global_pool, size=n_global, replace=True))
        n_hot = int(n_events * HOT_FRAC)
        session_items.extend(item_ids_arr[rng.choice(n_items, size=n_hot, p=item_probs, replace=True)])
        n_cluster = int(n_events * CLUSTER_FRAC)
        cpool = list(cluster_item_pool[uc])
        session_items.extend(rng.choice(cpool, size=n_cluster, replace=True))
        n_orig = n_events - n_global - n_hot - n_cluster
        if len(user_orig) > 0:
            session_items.extend(rng.choice(user_orig, size=n_orig, replace=True))
        else:
            session_items.extend(rng.choice(item_ids_arr, size=n_orig, replace=True))
        rng.shuffle(session_items)

        et = [session_base]
        for _ in range(n_events - 1):
            et.append(et[-1] + np.timedelta64(rng.randint(30, 300), 's'))

        for item_id, evt in zip(session_items, et):
            if item_id not in item_idx_map: continue
            idx = item_idx_map[item_id]
            raw_score = compute_raw(user_vec, idx)
            score = float(sigmoid(8.0 * raw_score))
            rec = {'user_id': uid, 'item_id': item_id, 'match_score': score,
                   'event_time_ms': int(evt.astype('datetime64[ms]').astype(np.int64)),
                   'cluster': int(uc)}
            is_c = item_is_commerce[idx]

            for behavior, (b, shift_arr, commerce_only) in BEHAVIOR_SPEC.items():
                if commerce_only and not is_c:
                    rec[behavior] = 0; continue
                s = shift_arr[idx]
                if behavior == 'comment':
                    s = s * social
                p = sigmoid(A_COMMON * (raw_score + s) - b)
                rec[behavior] = int(rng.random() < p)
            records.append(rec)

    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{len(user_latent)} users ({len(records):,} records)')

interactions_enriched = pd.DataFrame(records)
print(f'Generated {len(interactions_enriched):,} interactions')
for col in ['view','like','long_view','fav','comment','share','buy']:
    print(f'  {col}: {interactions_enriched[col].mean()*100:.1f}%')

# ============================================================
# 6. Validation
# ============================================================
print('\n=== Validation ===')
times = pd.to_datetime(interactions_enriched['event_time_ms'], unit='ms')
print(f'Time range: {times.min()} ~ {times.max()}')

iu_new = interactions_enriched.groupby('user_id').size()
print(f'Interactions/user: mean={iu_new.mean():.1f}')

# Jaccard
user_items = defaultdict(set)
for uid, iid in zip(interactions_enriched['user_id'], interactions_enriched['item_id']):
    user_items[uid].add(iid)
su = np.random.choice(list(user_items.keys()), size=min(500, len(user_items)), replace=False)
jaccards = []
for a in range(len(su)):
    for b in range(a+1, min(a+50, len(su))):
        s1, s2 = user_items[su[a]], user_items[su[b]]
        u = len(s1 | s2)
        if u > 0: jaccards.append(len(s1 & s2) / u)
ja = np.array(jaccards)
print(f'Jaccard: mean={ja.mean():.4f}, >5%={(ja>0.05).mean()*100:.1f}%')

# Top item share
item_freq = interactions_enriched['item_id'].value_counts()
top5_share = item_freq.head(int(n_items*0.05)).sum() / len(interactions_enriched)
print(f'Top 5% item share: {top5_share*100:.1f}%')

# Sessions
sc = interactions_enriched.groupby('user_id')['event_time_ms'].apply(
    lambda x: (x.sort_values().diff() > 3_600_000).sum() + 1)
print(f'Sessions/user: mean={sc.mean():.1f}')

# Behavior rates + AUC
behaviors = ['view','like','long_view','fav','comment','share','buy']
print(f'\nBehavior rates:')
for b in behaviors:
    print(f'  {b}: {interactions_enriched[b].mean()*100:.1f}%')
eng = (interactions_enriched[['like','long_view','fav','comment','share']].max(axis=1))
strong = (interactions_enriched[['like','fav','comment','share']].max(axis=1))
print(f'  engagement: {eng.mean()*100:.1f}%  strong_action: {strong.mean()*100:.1f}%')
print(f'  engagement != strong: {(eng != strong).mean()*100:.1f}%')

print(f'\nAUC (match_score → behavior):')
for b in behaviors:
    if interactions_enriched[b].nunique() > 1:
        auc = roc_auc_score(interactions_enriched[b], interactions_enriched['match_score'])
        print(f'  ms→{b:12s}: {auc:.4f}')
print(f'  ms→engagement:    {roc_auc_score(eng, interactions_enriched["match_score"]):.4f}')
print(f'  ms→strong_action: {roc_auc_score(strong, interactions_enriched["match_score"]):.4f}')

# Binary correlations
print(f'\nBinary correlations:')
for b1 in ['like','long_view','fav','comment','share']:
    row = [f'{b1:10s}'] + [f'{np.corrcoef(interactions_enriched[b1], interactions_enriched[b2])[0,1]:.3f}' for b2 in ['like','long_view','fav','comment','share']]
    print(' '.join(row))

# ============================================================
# 7. Interaction features + Save
# ============================================================
print('\n=== Computing interaction features ===')

user_stats = interactions_enriched.groupby('user_id').agg(
    user_n_interact=('item_id', 'size'),
    user_avg_ms=('match_score', 'mean'),
    user_std_ms=('match_score', 'std'),
    user_like_rate=('like', 'mean'),
    user_long_rate=('long_view', 'mean'),
    user_fav_rate=('fav', 'mean'),
    user_comment_rate=('comment', 'mean'),
    user_share_rate=('share', 'mean'),
    user_buy_rate=('buy', 'mean'),
).fillna(0)

item_stats = interactions_enriched.groupby('item_id').agg(
    item_n_interact=('user_id', 'size'),
    item_avg_ms=('match_score', 'mean'),
    item_like_rate=('like', 'mean'),
    item_long_rate=('long_view', 'mean'),
    item_fav_rate=('fav', 'mean'),
    item_comment_rate=('comment', 'mean'),
    item_share_rate=('share', 'mean'),
    item_buy_rate=('buy', 'mean'),
).fillna(0)

user_latent = user_latent.join(user_stats, how='left').fillna(0)
item_latent = item_latent.join(item_stats, how='left').fillna(0)

print(f'User features: {len(user_latent.columns)} columns (+{len(user_stats.columns)} interaction)')
print(f'Item features: {len(item_latent.columns)} columns (+{len(item_stats.columns)} interaction)')
print(f'  user_n_interact: mean={user_stats["user_n_interact"].mean():.0f}, max={user_stats["user_n_interact"].max():.0f}')
print(f'  item_n_interact: mean={item_stats["item_n_interact"].mean():.0f}, max={item_stats["item_n_interact"].max():.0f}')

interactions_enriched.to_csv(f'{DATA_DIR}/interactions_enriched.csv', index=False)
user_latent.to_csv(f'{DATA_DIR}/user_latent.csv', index_label='user_id')
item_latent.to_csv(f'{DATA_DIR}/item_latent.csv', index_label='item_id')
user_latent[['cluster']].to_csv(f'{DATA_DIR}/user_clusters.csv')

print(f'\n=== Phase B Complete ===')
print(f'Interactions: {len(interactions_enriched):,}')
print(f'Time range: {times.min()} ~ {times.max()}')
print(f'Jaccard: {ja.mean():.4f} (>5%: {(ja>0.05).mean()*100:.1f}%)')
print(f'User features: {len(user_latent.columns)} | Item features: {len(item_latent.columns)}')
print('Next: Run Phase C (notebook or run_phase_c.py)')
