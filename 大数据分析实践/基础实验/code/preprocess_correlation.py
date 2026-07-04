"""
数据预处理方案验证：输出 per-36 特征的相关性矩阵
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'NBA_Season_Stats.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'feature_correlation_matrix.csv')

# ======================== 1. 加载 ========================
df = pd.read_csv(DATA_PATH)
print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

# ======================== 2. 构建衍生字段 ========================

# 2a. eFG% 补充计算
df['eFG%'] = df['eFG%'].fillna((df['FG'] + 0.5 * df['3P']) / df['FGA'])

# 2b. Per-36 指标（需要先处理 MP=0）
df['MP'] = df['MP'].replace(0, np.nan)

per36_features = {
    'PTS_per36': 'PTS',
    'TRB_per36': 'TRB',
    'AST_per36': 'AST',
    'STL_per36': 'STL',
    'BLK_per36': 'BLK',
    'PF_per36': 'PF',
    'TOV_per36': 'TOV',
    '3PA_per36': '3PA',
    'FTA_per36': 'FTA',
}
for new_f, raw_f in per36_features.items():
    df[new_f] = df[raw_f] / df['MP'] * 36

# 2c. 助攻失误比
df['AST_TOV'] = df['AST'] / df['TOV'].replace(0, np.nan)
# TOV=0 的情况用最大值替代（取所有有限值的最大值）
max_ast_tov = df.loc[np.isfinite(df['AST_TOV']), 'AST_TOV'].max()
df['AST_TOV'] = df['AST_TOV'].fillna(max_ast_tov)

# 2d. 场均上场时间
df['MPG'] = df['MP'] / df['G']

# ======================== 3. 缺失值处理 ========================
fill_zero_cols = ['FG%', '3P%', 'FT%', '2P%', '2PA', 'eFG%']
for col in fill_zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print(f"\n缺失值处理完成")

# ======================== 4. 行过滤 ========================
n_before = len(df)
df = df[df['MP'].notna()]            # MP > 0
df = df[df['MP'] >= 200]             # 上场时间足够长
df = df[df['G'] >= 10]               # 出场次数足够多
df = df[df['Tm'] != 'TOT']           # 剔除多队合计行
print(f"行过滤: {n_before} -> {len(df)} 条 (剔除 {n_before - len(df)} 条)")

# ======================== 5. 特征集 ========================
FEATURES = [
    'PTS_per36', 'TRB_per36', 'AST_per36', 'STL_per36', 'BLK_per36',
    '3P%', 'FT%', 'eFG%',
    '3PA_per36', 'FTA_per36', 'TOV_per36', 'PF_per36',
]

# 验证特征无缺失
missing = df[FEATURES].isnull().sum()
if missing.sum() > 0:
    print(f"\n[警告] 特征中存在缺失值:")
    print(missing[missing > 0])
else:
    print(f"\n[OK] 所有特征无缺失值")

# ======================== 6. 相关性矩阵 ========================
print(f"\n{'='*60}")
print(f"特征相关性矩阵（{len(FEATURES)} × {len(FEATURES)}）")
print(f"{'='*60}")

corr = df[FEATURES].corr()

# 打印到控制台（格式化的表格）
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', '{:.3f}'.format)
print(f"\n{corr.to_string()}")
print(f"\n矩阵形状: {corr.shape}")

# ======================== 7. 保存 CSV ========================
corr.to_csv(OUTPUT_PATH, encoding='utf-8-sig')
print(f"\n已保存: {OUTPUT_PATH}")

# ======================== 8. 强相关性摘要 ========================
print(f"\n{'='*60}")
print(f"强相关性摘要 (|r| >= 0.7)")
print(f"{'='*60}")
strong_pairs = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.7:
            strong_pairs.append((FEATURES[i], FEATURES[j], r))

if strong_pairs:
    for f1, f2, r in sorted(strong_pairs, key=lambda x: -abs(x[2])):
        direction = "+" if r > 0 else "-"
        print(f"  {f1}  <->  {f2}   =  {direction}{abs(r):.3f}")
else:
    print("  无 |r| >= 0.7 的强相关对")

# 近零相关摘要
print(f"\n近零相关摘要 (|r| < 0.1)")
near_zero_pairs = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = corr.iloc[i, j]
        if abs(r) < 0.1:
            near_zero_pairs.append((FEATURES[i], FEATURES[j], r))

if near_zero_pairs:
    for f1, f2, r in sorted(near_zero_pairs, key=lambda x: abs(x[2])):
        print(f"  {f1}  <->  {f2}   =  {r:+.3f}")
else:
    print("  无 |r| < 0.1 的近零相关对")
