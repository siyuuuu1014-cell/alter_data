"""
Phase C standalone runner — KuaiRand 7-file format generation.
Uses long_view directly from Phase B (no longer derived from fav+like).
"""
import pandas as pd, numpy as np
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
OUT_DIR = 'data/kuairand'
SRC_DIR = 'D:/dev/projects/zhere_olap/data/recommendation'
SEED = 42
np.random.seed(SEED)
rng = np.random.RandomState(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
print('=== Loading data ===')
user_latent = pd.read_csv(f'{DATA_DIR}/user_latent.csv', index_col='user_id')
item_latent = pd.read_csv(f'{DATA_DIR}/item_latent.csv', index_col='item_id')
interactions = pd.read_csv(f'{DATA_DIR}/interactions_enriched.csv')

# Fix double prefix
rename_map = {}
for c in user_latent.columns:
    if c.startswith('theme_theme_'):
        rename_map[c] = c.replace('theme_theme_', 'theme_')
user_latent.rename(columns=rename_map, inplace=True)

print(f'Users: {len(user_latent):,}, Items: {len(item_latent):,}, Interactions: {len(interactions):,}')

# Load text lookup
asset_df = pd.read_csv(f'{SRC_DIR}/published_asset.csv')
prompt_df = pd.read_csv(f'{SRC_DIR}/published_prompt.csv')
commerce_df = pd.read_csv(f'{SRC_DIR}/published_commerce.csv')

text_lookup = {}
for _, row in asset_df.iterrows():
    text_lookup[row['asset_id']] = {
        'title': str(row.get('title','') or ''),
        'description': str(row.get('description','') or ''),
        'theme': str(row.get('theme','') or ''),
        'location': str(row.get('location','') or ''),
        'content_type': 'asset',
        'duration': int(row.get('duration',0) or 0),
    }
for _, row in prompt_df.iterrows():
    text_lookup[row['prompt_id']] = {
        'title': str(row.get('title','') or ''),
        'description': str(row.get('description','') or ''),
        'theme': str(row.get('theme','') or ''),
        'location': str(row.get('location','') or ''),
        'content_type': 'prompt',
        'duration': 0,
    }
for _, row in commerce_df.iterrows():
    desc = ' '.join(str(row.get(f,'') or '') for f in ['organization','activity','context','requirements'])
    text_lookup[row['commerce_id']] = {
        'title': str(row.get('activity','') or ''),
        'description': desc,
        'theme': str(row.get('range','') or ''),
        'location': str(row.get('place','') or ''),
        'content_type': 'commerce',
        'duration': 0,
    }
print(f'Text lookup: {len(text_lookup):,} items')

# ============================================================
# 1. user_features_pure.csv
# ============================================================
print('\n=== 1. user_features_pure.csv ===')
user_exclude = ['cluster']
user_feature_cols = [c for c in user_latent.columns if c not in user_exclude]
user_features = user_latent[user_feature_cols].reset_index().fillna(0)

iu = interactions.groupby('user_id').size()
user_features['user_active_degree'] = user_features['user_id'].map(
    lambda u: min(iu.get(u, 0) / 80, 1.0))

user_features.to_csv(f'{OUT_DIR}/user_features_pure.csv', index=False)
print(f'Saved: {len(user_features)} users x {len(user_features.columns)} features')

# ============================================================
# 2. video_features_basic_pure.csv
# ============================================================
print('\n=== 2. video_features_basic_pure.csv ===')
CT_MAP = {'asset': 1, 'prompt': 2, 'commerce': 3}
item_exclude = ['cluster']
video_feature_cols = [c for c in item_latent.columns if c not in item_exclude]
video_features = item_latent[video_feature_cols].reset_index().fillna(0)
video_features = video_features.rename(columns={'item_id': 'video_id'})

ct_cols = ['ct_published_asset', 'ct_published_prompt', 'ct_published_commerce']
for col in ct_cols:
    if col in video_features.columns:
        ct_type = col.replace('ct_published_', '')
        video_features.loc[video_features[col] == 1, 'video_type'] = CT_MAP.get(ct_type, 0)
video_features['video_type'] = video_features['video_type'].fillna(0).astype(int)

video_features.to_csv(f'{OUT_DIR}/video_features_basic_pure.csv', index=False)
print(f'Saved: {len(video_features)} videos x {len(video_features.columns)} features')
print(f'video_type dist: {video_features["video_type"].value_counts().to_dict()}')

# ============================================================
# 3-5. Time split: Train/Val/Test
# ============================================================
print('\n=== 3. Time split ===')
times = pd.to_datetime(interactions['event_time_ms'], unit='ms')
interactions['timestamp'] = times
interactions['date'] = times.dt.strftime('%Y-%m-%d')

t_min, t_max = times.min(), times.max()
t_range = (t_max - t_min).total_seconds()
train_cut = t_min + pd.Timedelta(seconds=t_range * 0.70)
val_cut   = t_min + pd.Timedelta(seconds=t_range * 0.85)

print(f'Time: {t_min} ~ {t_max}')
print(f'Train cutoff: {train_cut}')
print(f'Val cutoff:   {val_cut}')

train_mask = times <= train_cut
val_mask   = (times > train_cut) & (times <= val_cut)
test_mask  = times > val_cut

print(f'Train: {train_mask.sum():,} ({train_mask.mean()*100:.1f}%)')
print(f'Val:   {val_mask.sum():,} ({val_mask.mean()*100:.1f}%)')
print(f'Test:  {test_mask.sum():,} ({test_mask.mean()*100:.1f}%)')

# ============================================================
# Build log function (Plan C: long_view from Phase B)
# ============================================================
def build_log(df_split):
    log = pd.DataFrame()
    log['user_id']   = df_split['user_id']
    log['video_id']  = df_split['item_id']
    log['date']      = df_split['date']
    log['time_ms']   = df_split['event_time_ms']
    log['is_click']  = df_split['view'].astype(int)
    log['is_like']   = df_split['like'].astype(int)
    log['is_follow'] = df_split['fav'].astype(int)
    log['is_comment']= df_split['comment'].astype(int)
    log['is_forward']= df_split['share'].astype(int)
    log['is_hate']   = 0
    # Plan C: long_view is an independent behavior from Phase B
    log['long_view'] = df_split['long_view'].astype(int)
    log['play_time_ms'] = (df_split['match_score'] * 120000).astype(int)
    log['duration_ms']  = df_split['item_id'].map(
        lambda iid: text_lookup.get(iid, {}).get('duration', 60) * 1000).fillna(60000).astype(int)
    log['profile_stay_time'] = 0
    log['comment_stay_time'] = df_split['comment'].apply(lambda x: rng.randint(5000,30000) if x==1 else 0)
    return log

log_train = build_log(interactions[train_mask])
log_val   = build_log(interactions[val_mask])
log_test  = build_log(interactions[test_mask])

log_train.to_csv(f'{OUT_DIR}/log_standard_train_pure.csv', index=False)
log_val.to_csv(  f'{OUT_DIR}/log_standard_val_pure.csv',   index=False)
log_test.to_csv( f'{OUT_DIR}/log_standard_test_pure.csv',  index=False)

print(f'Saved: train={len(log_train):,}, val={len(log_val):,}, test={len(log_test):,}')
print(f'long_view (independent): train={log_train["long_view"].mean()*100:.1f}% '
      f'val={log_val["long_view"].mean()*100:.1f}% test={log_test["long_view"].mean()*100:.1f}%')

# ============================================================
# 6. log_random_test_pure.csv
# ============================================================
print('\n=== 6. log_random_test_pure.csv ===')
all_item_ids = item_latent.index.values
all_user_ids = user_latent.index.values

# --- Pre-compute item-level implicit engagement probability ---
# Real KuaiRand random log has ~17% pos_rate from implicit feedback (play_time → long_view).
# We model this as: p_long = sigmoid(quality * 5 + popularity * 3 - 4.8) → ~17% with feature dependence.
item_quality_raw = item_latent['quality_score'].values
item_pop_raw = item_latent['popularity'].values
ITEM_IMPLICIT_LOGIT = 5.0 * item_quality_raw + 3.0 * item_pop_raw - 6.2
ITEM_IMPLICIT_P_LONG = 1.0 / (1.0 + np.exp(-ITEM_IMPLICIT_LOGIT))
print(f'Item implicit p_long: mean={ITEM_IMPLICIT_P_LONG.mean():.3f}, '
      f'range=[{ITEM_IMPLICIT_P_LONG.min():.3f}, {ITEM_IMPLICIT_P_LONG.max():.3f}]')

random_records = []
test_start = train_cut + pd.Timedelta(seconds=1)

for uid in all_user_ids:
    n_random = rng.randint(15, 26)
    random_items = rng.choice(all_item_ids, size=n_random, replace=False)
    test_end = t_max
    test_secs = (test_end - test_start).total_seconds()
    random_offsets = rng.randint(0, int(test_secs), size=n_random)
    random_times = [test_start + pd.Timedelta(seconds=int(s)) for s in random_offsets]

    for iid, ts in zip(random_items, random_times):
        # Map item UUID to positional index
        iid_idx = item_latent.index.get_loc(iid)
        p_long = ITEM_IMPLICIT_P_LONG[iid_idx]
        has_long = int(rng.random() < p_long)
        # play_time: if long_view, 90-150s; else 0-30s
        if has_long:
            play_time = rng.randint(90000, 150000)
        else:
            play_time = rng.randint(0, 30000)
        random_records.append({
            'user_id': uid, 'video_id': iid,
            'date': ts.strftime('%Y-%m-%d'),
            'time_ms': int(ts.timestamp() * 1000),
            'is_click': 0, 'is_like': 0, 'is_follow': 0,
            'is_comment': 0, 'is_forward': 0, 'is_hate': 0,
            'long_view': has_long, 'play_time_ms': play_time,
            'duration_ms': text_lookup.get(iid, {}).get('duration', 60) * 1000,
            'profile_stay_time': 0, 'comment_stay_time': 0,
            'is_random': 1,
        })

log_random = pd.DataFrame(random_records)
log_random.to_csv(f'{OUT_DIR}/log_random_test_pure.csv', index=False)
print(f'Saved: {len(log_random):,} random exposures ({len(log_random)/len(all_user_ids):.1f}/user)')

# ============================================================
# 7. kuairand_video_captions.csv
# ============================================================
print('\n=== 7. kuairand_video_captions.csv ===')
captions = []
for iid in item_latent.index:
    info = text_lookup.get(iid, {})
    caption_text = f"{info.get('title','')} {info.get('description','')}".strip()[:512]
    cover_text = f"{info.get('theme','')} {info.get('location','')}".strip()[:256]
    captions.append({'video_id': iid, 'caption_text': caption_text, 'cover_text': cover_text})

captions_df = pd.DataFrame(captions)
captions_df.to_csv(f'{OUT_DIR}/kuairand_video_captions.csv', index=False)
print(f'Saved: {len(captions_df):,} captions')
print(f'Non-empty: {(captions_df["caption_text"].str.len()>0).sum():,}')

# ============================================================
# 8. kuairand_video_categories.csv
# ============================================================
print('\n=== 8. kuairand_video_categories.csv ===')
categories = []
for iid in item_latent.index:
    info = text_lookup.get(iid, {})
    level1 = info.get('content_type', 'unknown').capitalize()
    level2 = info.get('theme', '') or info.get('location', '') or 'Other'
    categories.append({'video_id': iid, 'category_level_1': level1, 'category_level_2': level2})

categories_df = pd.DataFrame(categories)
categories_df.to_csv(f'{OUT_DIR}/kuairand_video_categories.csv', index=False)

print(f'Saved: {len(categories_df):,} categories')
print(f'Level 1: {categories_df["category_level_1"].value_counts().to_dict()}')
print(f'Level 2 unique: {categories_df["category_level_2"].nunique()}')

# ============================================================
# Verification
# ============================================================
print('\n' + '='*60)
print('Phase C Output Summary')
print('='*60)
files = [
    ('user_features_pure.csv', user_features),
    ('video_features_basic_pure.csv', video_features),
    ('log_standard_train_pure.csv', log_train),
    ('log_standard_val_pure.csv', log_val),
    ('log_standard_test_pure.csv', log_test),
    ('log_random_test_pure.csv', log_random),
    ('kuairand_video_captions.csv', captions_df),
    ('kuairand_video_categories.csv', categories_df),
]
for name, df in files:
    path = f'{OUT_DIR}/{name}'
    size_mb = os.path.getsize(path) / 1024**2
    print(f'  {name:<38} {len(df):>8,} rows x {len(df.columns):>3} cols  [{size_mb:.1f} MB]')

print(f'\nTrain/Val/Test: {len(log_train):,} / {len(log_val):,} / {len(log_test):,}')

for label, log in [('train', log_train), ('val', log_val), ('test', log_test)]:
    print(f'{label}: click={log["is_click"].mean()*100:.1f}% like={log["is_like"].mean()*100:.1f}% '
          f'follow={log["is_follow"].mean()*100:.1f}% comment={log["is_comment"].mean()*100:.1f}% '
          f'forward={log["is_forward"].mean()*100:.1f}% long_view={log["long_view"].mean()*100:.1f}%')

print(f'\nRandom log: {len(log_random):,} exposures')
print(f'Time span: {log_train["date"].min()} -> {log_test["date"].max()}')
print(f'\nPhase C Complete! Next: adapt_for_sfg.py + Phase D training')
