"""
Phase D - Data Adaptation Script
Converts our KuaiRand files to the format expected by the SFG-BiCross notebook.

Changes:
  1. user_id / video_id: UUID string → sequential int64
  2. date: "2026-06-23" → 20260623 (int)
  3. File naming: log_standard_{train,val,test}_pure.csv → expected names

Run once before training:
  python adapt_for_sfg.py
"""
import pandas as pd
import numpy as np
import os, shutil

SRC = 'data/kuairand'
DST = 'data/kuairand_adapted'
SEED = 42
np.random.seed(SEED)

os.makedirs(DST, exist_ok=True)
print(f'Adapting data: {SRC} → {DST}\n')

# ============================================================
# 1. Build ID mappings (UUID → int)
# ============================================================
print('=== 1. Building ID mappings ===')

user_features = pd.read_csv(f'{SRC}/user_features_pure.csv')
video_features = pd.read_csv(f'{SRC}/video_features_basic_pure.csv')

# user_id mapping
user_id_map = {uid: i for i, uid in enumerate(user_features['user_id'])}
user_id_df = pd.DataFrame({'user_id_orig': list(user_id_map.keys()), 'user_id_int': list(user_id_map.values())})
user_id_df.to_csv(f'{DST}/user_id_mapping.csv', index=False)
print(f'User mapping: {len(user_id_map):,} UUIDs → 0..{len(user_id_map)-1}')

# video_id mapping
video_id_map = {vid: i for i, vid in enumerate(video_features['video_id'])}
video_id_df = pd.DataFrame({'video_id_orig': list(video_id_map.keys()), 'video_id_int': list(video_id_map.values())})
video_id_df.to_csv(f'{DST}/video_id_mapping.csv', index=False)
print(f'Video mapping: {len(video_id_map):,} UUIDs → 0..{len(video_id_map)-1}')

# ============================================================
# 2. Pre-compute per-video aggregates from logs
#    (needed for video_features and captions enrichment)
# ============================================================
print('\n=== 2. Pre-computing per-video stats ===')

all_logs_raw = pd.concat([
    pd.read_csv(f'{SRC}/{f}', low_memory=False)
    for f in ['log_standard_train_pure.csv', 'log_standard_val_pure.csv',
              'log_random_test_pure.csv', 'log_standard_test_pure.csv']
], ignore_index=True)

# Per-video: earliest event as proxy for upload time
all_logs_raw['video_id_int'] = all_logs_raw['video_id'].map(video_id_map)
all_logs_raw['datetime_str'] = all_logs_raw['date'] + ' ' + (
    pd.to_datetime(all_logs_raw['time_ms'], unit='ms').dt.strftime('%H:%M:%S')
)
# Get first event timestamp per video → upload_dt
first_event = all_logs_raw.groupby('video_id_int')['time_ms'].min().reset_index()
first_event['upload_dt'] = pd.to_datetime(first_event['time_ms'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
# Subtract random offset (1-30 days before first event) so upload predates first interaction
first_event['upload_dt'] = pd.to_datetime(first_event['upload_dt']) - pd.to_timedelta(
    np.random.randint(1, 31, size=len(first_event)), unit='D'
)
vid_upload = first_event.set_index('video_id_int')['upload_dt'].to_dict()

# Per-video: median duration (for video_duration in features + captions)
dur_mask = all_logs_raw['duration_ms'] > 0
vid_dur = all_logs_raw[dur_mask].groupby('video_id_int')['duration_ms'].median().to_dict()
print(f'  upload_dt coverage: {len(vid_upload):,}/{len(video_id_map):,} videos')
print(f'  duration_ms coverage: {len(vid_dur):,}/{len(video_id_map):,} videos')

# ============================================================
# 3. Adapt user_features_pure.csv
#    Add missing KuaiRand columns: is_lowactive_period, is_live_streamer,
#    is_video_author, follow/fans/friend nums, register_days, onehot_feat*
#    Rewrite user_active_degree: float → categorical string
# ============================================================
print('\n=== 3. Adapting user_features ===')
uf = user_features.copy()
uf['user_id'] = uf['user_id'].map(user_id_map)
assert uf['user_id'].notna().all(), 'Failed to map some user_ids!'
uf['user_id'] = uf['user_id'].astype('int64')

# --- Compute per-user stats from logs for feature derivation ---
raw_logs = pd.concat([
    pd.read_csv(f'{SRC}/{f}', low_memory=False)
    for f in ['log_standard_train_pure.csv']
], ignore_index=True)
raw_logs['user_id_int'] = raw_logs['user_id'].map(user_id_map)
user_interact = raw_logs.groupby('user_id_int').agg(
    n_interact=('video_id', 'size'),
    n_likes=('is_like', 'sum'),
).reset_index()
user_interact = user_interact.set_index('user_id_int')

# --- Rewrite user_active_degree: float → categorical for KuaiRand ---
#   CFG.ACTIVE_DEGREES = ["high_active", "full_active"] → need ~50% of users
n_interact = uf['user_id'].map(user_interact['n_interact']).fillna(0)
try:
    uf['user_active_degree'] = pd.qcut(
        n_interact, q=[0, 0.05, 0.20, 0.50, 0.80, 1.00],
        labels=['new', 'low_active', 'middle_active', 'high_active', 'full_active'],
        duplicates='drop',
    )
except Exception:
    # Fallback: simple cut
    med = n_interact.median()
    uf['user_active_degree'] = pd.cut(
        n_interact,
        bins=[0, max(10, n_interact.quantile(0.05)), max(30, n_interact.quantile(0.20)),
              med, n_interact.quantile(0.80), 999],
        labels=['new', 'low_active', 'middle_active', 'high_active', 'full_active'],
    )

# --- is_lowactive_period: 0 for all (must be 0 to pass active_mask) ---
uf['is_lowactive_period'] = 0

# --- is_live_streamer / is_video_author: small fraction (5-10%) ---
rng_user = np.random.RandomState(SEED + 42)
uf['is_live_streamer'] = (rng_user.rand(len(uf)) < 0.05).astype('int64')
uf['is_video_author']  = (rng_user.rand(len(uf)) < 0.10).astype('int64')

# --- Social media style numeric features ---
uf['follow_user_num'] = np.clip(rng_user.lognormal(mean=2.5, sigma=1.0, size=len(uf)), 0, 5000).astype('int64')
uf['fans_user_num']   = np.clip(rng_user.lognormal(mean=2.0, sigma=1.2, size=len(uf)), 0, 10000).astype('int64')
uf['friend_user_num'] = np.clip(rng_user.lognormal(mean=3.0, sigma=0.8, size=len(uf)), 0, 3000).astype('int64')
uf['register_days']   = np.clip(rng_user.randint(1, 1500, size=len(uf)), 1, 2000).astype('int64')

# --- Range categorical versions ---
def to_range(val, edges):
    for i in range(len(edges) - 1):
        if val < edges[i+1]:
            return f'{edges[i]}~{edges[i+1]}'
    return str(edges[-1])
follow_edges = [0, 3, 10, 30, 100, 5000]
fans_edges = [0, 1, 5, 20, 50, 10000]
friend_edges = [0, 1, 5, 10, 30, 3000]
reg_edges = [0, 30, 90, 365, 730, 2000]
uf['follow_user_num_range'] = uf['follow_user_num'].apply(lambda x: to_range(x, follow_edges))
uf['fans_user_num_range']   = uf['fans_user_num'].apply(lambda x: to_range(x, fans_edges))
uf['friend_user_num_range'] = uf['friend_user_num'].apply(lambda x: to_range(x, friend_edges))
uf['register_days_range']   = uf['register_days'].apply(lambda x: to_range(x, reg_edges))

# --- onehot_feat0..17: random 0/1 features ---
for i in range(18):
    uf[f'onehot_feat{i}'] = rng_user.randint(0, 2, size=len(uf))

uf.to_csv(f'{DST}/user_features_pure.csv', index=False)
print(f'Saved: {len(uf)} users × {len(uf.columns)} features')
print(f'  user_active_degree: {uf["user_active_degree"].value_counts().to_dict()}')
print(f'  is_lowactive_period={uf["is_lowactive_period"].nunique()-1} (all 0), '
      f'is_live_streamer={uf["is_live_streamer"].sum()}, is_video_author={uf["is_video_author"].sum()}')

# ============================================================
# 4. Adapt video_features_basic_pure.csv
#    Notebook expects extra columns: upload_dt, video_duration,
#    server_width, server_height, upload_type, visible_status, music_type
# ============================================================
print('\n=== 4. Adapting video_features ===')
vf = video_features.copy()
vf['video_id'] = vf['video_id'].map(video_id_map)
assert vf['video_id'].notna().all(), 'Failed to map some video_ids!'
vf['video_id'] = vf['video_id'].astype('int64')

# --- Add missing columns expected by notebook Cell 11 ---

# upload_dt: derived from first event per video (proxy for when video was uploaded)
vf['upload_dt'] = vf['video_id'].map(vid_upload)
# Fill missing with a default date in the data range
default_dt = pd.Timestamp('2026-03-01 00:00:00')
vf['upload_dt'] = vf['upload_dt'].fillna(default_dt).dt.strftime('%Y-%m-%d %H:%M:%S')

# video_duration: same as duration from logs (milliseconds)
vf['video_duration'] = vf['video_id'].map(vid_dur).fillna(60000).astype('int64')

# server_width / server_height: by video_type (1=Asset→1920×1080, 2=Prompt→1080×1920, 3=Commerce→1080×1080)
width_map  = {1: 1920, 2: 1080, 3: 1080}
height_map = {1: 1080, 2: 1920, 3: 1080}
vf['server_width']  = vf['video_type'].map(width_map).fillna(1080).astype('int64')
vf['server_height'] = vf['video_type'].map(height_map).fillna(1080).astype('int64')
# Add small variance (±10%)
rng_w = np.random.RandomState(SEED)
rng_h = np.random.RandomState(SEED + 1)
vf['server_width']  = (vf['server_width']  * (0.9 + 0.2 * rng_w.rand(len(vf)))).astype('int64')
vf['server_height'] = (vf['server_height'] * (0.9 + 0.2 * rng_h.rand(len(vf)))).astype('int64')

# upload_type, visible_status, music_type: default values
vf['upload_type'] = 0
vf['visible_status'] = 0
vf['music_type'] = 0

# tag: empty string (parse_tags returns [] for NaN/empty, no tags in synthetic data)
vf['tag'] = ''

vf.to_csv(f'{DST}/video_features_basic_pure.csv', index=False)
print(f'Saved: {len(vf)} videos × {len(vf.columns)} features')
print(f'  New columns: upload_dt, video_duration, server_width, server_height, upload_type, visible_status, music_type')

# ============================================================
# 5. Adapt log files
# ============================================================
print('\n=== 5. Adapting log files ===')

def adapt_log(fname_in, fname_out):
    log = pd.read_csv(f'{SRC}/{fname_in}', low_memory=False)
    log['user_id'] = log['user_id'].map(user_id_map).astype('int64')
    log['video_id'] = log['video_id'].map(video_id_map).astype('int64')
    # Convert date: "2026-06-23" → 20260623
    log['date'] = pd.to_datetime(log['date']).dt.strftime('%Y%m%d').astype('int64')
    log.to_csv(f'{DST}/{fname_out}', index=False)
    print(f'  {fname_in} → {fname_out}: {len(log):,} rows')
    return log

train = adapt_log('log_standard_train_pure.csv', 'log_standard_4_08_to_4_21_pure.csv')
val   = adapt_log('log_standard_val_pure.csv',   'log_standard_4_22_to_5_08_pure.csv')
test  = adapt_log('log_random_test_pure.csv',    'log_random_4_22_to_5_08_pure.csv')
# Also keep standard test log
adapt_log('log_standard_test_pure.csv', 'log_standard_test_pure.csv')

# ============================================================
# 6. Adapt captions file
#    notebook expects: final_video_id, caption, show_cover_text, duration
#    we have:          video_id,       caption_text, cover_text
# ============================================================
print('\n=== 6. Adapting video captions ===')
captions = pd.read_csv(f'{SRC}/kuairand_video_captions.csv')
captions['video_id'] = captions['video_id'].map(video_id_map).astype('int64')
captions = captions.rename(columns={
    'video_id': 'final_video_id',
    'caption_text': 'caption',
    'cover_text': 'show_cover_text',
})

# Use pre-computed per-video durations
captions['duration'] = captions['final_video_id'].map(vid_dur)
captions['duration'] = captions['duration'].fillna(60000).astype('int64')

captions.to_csv(f'{DST}/kuairand_video_captions.csv', index=False)
print(f'  kuairand_video_captions.csv: {len(captions):,} rows × {len(captions.columns)} cols')
print(f'  Duration - min={captions["duration"].min()}, median={captions["duration"].median():.0f}, max={captions["duration"].max()}')

# ============================================================
# 7. Adapt categories file
#    notebook expects hierarchical: first/second/third/fourth_level_category_{id,name,prob}
#    we have: video_id, category_level_1, category_level_2
# ============================================================
print('\n=== 7. Adapting video categories ===')
cats = pd.read_csv(f'{SRC}/kuairand_video_categories.csv')
cats['video_id'] = cats['video_id'].map(video_id_map).astype('int64')
cats = cats.rename(columns={'video_id': 'final_video_id'})

# Encode category names → integer IDs
l1_unique = sorted(cats['category_level_1'].dropna().unique())
l2_unique = sorted(cats['category_level_2'].dropna().unique())
l1_to_id = {name: i+1 for i, name in enumerate(l1_unique)}
l2_to_id = {name: i+1 for i, name in enumerate(l2_unique)}

cats['first_level_category_name'] = cats['category_level_1'].fillna('Other')
cats['first_level_category_id'] = cats['first_level_category_name'].map(l1_to_id)
cats['first_level_category_prob'] = 1.0

cats['second_level_category_name'] = cats['category_level_2'].fillna('Other')
cats['second_level_category_id'] = cats['second_level_category_name'].map(l2_to_id)
cats['second_level_category_prob'] = 1.0

# Level 3 & 4: empty (our data has only 2 levels)
cats['third_level_category_id'] = 0
cats['third_level_category_name'] = ''
cats['third_level_category_prob'] = 0.0
cats['fourth_level_category_id'] = 0
cats['fourth_level_category_name'] = ''
cats['fourth_level_category_prob'] = 0.0

# Drop old columns, keep notebook-expected columns
CAT_COLS = [
    'final_video_id',
    'first_level_category_id', 'first_level_category_name', 'first_level_category_prob',
    'second_level_category_id', 'second_level_category_name', 'second_level_category_prob',
    'third_level_category_id', 'third_level_category_name', 'third_level_category_prob',
    'fourth_level_category_id', 'fourth_level_category_name', 'fourth_level_category_prob',
]
cats = cats[CAT_COLS]
cats.to_csv(f'{DST}/kuairand_video_categories.csv', index=False)
print(f'  kuairand_video_categories.csv: {len(cats):,} rows × {len(cats.columns)} cols')
print(f'  L1: {len(l1_unique)} categories (IDs 1..{len(l1_unique)})')
print(f'  L2: {len(l2_unique)} categories (IDs 1..{len(l2_unique)})')

# ============================================================
# 8. Verify
# ============================================================
print('\n=== 8. Verification ===')
for f in sorted(os.listdir(DST)):
    path = f'{DST}/{f}'
    mb = os.path.getsize(path) / 1024**2
    df = pd.read_csv(path, nrows=3)
    print(f'  {f:<45} {mb:>6.1f} MB  {len(df.columns):>3} cols')

# Quick check
tl = pd.read_csv(f'{DST}/log_standard_4_08_to_4_21_pure.csv')
print(f'\n  Train user_id range: [{tl["user_id"].min()}, {tl["user_id"].max()}]')
print(f'  Train video_id range: [{tl["video_id"].min()}, {tl["video_id"].max()}]')
print(f'  Train date range: [{tl["date"].min()}, {tl["date"].max()}]')
print(f'  Sample date: {tl["date"].iloc[0]} (type: {type(tl["date"].iloc[0]).__name__})')

# Verify UUID→int reversibility
reverse_uid = pd.read_csv(f'{DST}/user_id_mapping.csv')
print(f'\n  Reverse mapping check: '
      f'user_id_int 0 → {reverse_uid[reverse_uid["user_id_int"]==0]["user_id_orig"].iloc[0]}')

print('\n=== Data adaptation complete ===')
print(f'Adapted data saved to: {DST}/')
print('Ready for SFG-BiCross training.')
