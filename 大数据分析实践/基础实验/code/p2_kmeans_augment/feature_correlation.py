"""
KMeans 改进实验 — 特征相关性分析
===================================
预处理沿用 per-36 方案（TOT 去重、MP/G 过滤）
特征集：PTS_per36, TRB_per36, AST_per36, STL_per36, BLK_per36,
        TOV_per36, PF_per36, 3PAr, FTr
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'NBA_Season_Stats.csv')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
PIC_DIR = os.path.join(BASE_DIR, 'pics')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PIC_DIR, exist_ok=True)

# ======================== 1. 数据加载与预处理 ========================
print("1. 数据加载与预处理")
df = pd.read_csv(DATA_PATH)
print(f"  原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

# ---- Per-36 衍生 ----
per36_map = {
    'PTS_per36': 'PTS', 'TRB_per36': 'TRB', 'AST_per36': 'AST',
    'STL_per36': 'STL', 'BLK_per36': 'BLK',
    'TOV_per36': 'TOV', 'PF_per36': 'PF',
}
for new_f, raw_f in per36_map.items():
    df[new_f] = df[raw_f] / df['MP'].replace(0, np.nan) * 36

# ---- 3PAr 与 FTr ----
df['3PAr'] = df['3PA'] / df['FGA'].replace(0, np.nan)
df['FTr'] = df['FTA'] / df['FGA'].replace(0, np.nan)

# ---- 缺失值填充 ----
df['3PAr'] = df['3PAr'].fillna(0)
df['FTr'] = df['FTr'].fillna(0)

# ---- TOT 去重 ----
has_tot = df.groupby(['Year', 'Player'])['Tm'].transform(lambda x: (x == 'TOT').any())
df = df[~(has_tot & (df['Tm'] != 'TOT'))].copy()

# ---- 行过滤 ----
df = df[df['MP'] >= 200]
df = df[df['G'] >= 10]
print(f"  预处理后: {len(df)} 条")

# ======================== 2. 特征集 ========================
FEATURES = ['PTS_per36', 'TRB_per36', 'AST_per36', 'STL_per36',
            'BLK_per36', 'TOV_per36', 'PF_per36', '3PAr', 'FTr']
print(f"  特征: {FEATURES}")

# 验证无缺失
missing = df[FEATURES].isnull().sum()
if missing.sum() > 0:
    print(f"  [警告] 存在缺失值: {missing[missing > 0].to_dict()}")

# ======================== 3. 相关性矩阵 ========================
print("\n2. 相关性矩阵")
corr = df[FEATURES].corr()

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.4f}'.format)
print(f"\n{corr.to_string()}")

# 保存 CSV
corr.to_csv(os.path.join(RESULT_DIR, 'correlation_matrix.csv'), encoding='utf-8-sig')
print(f"\n  已保存: result/correlation_matrix.csv")

# ======================== 4. 强相关摘要 ========================
print(f"\n3. 强相关对 (|r| >= 0.7):")
strong = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.7:
            d = '+' if r > 0 else '-'
            strong.append((FEATURES[i], FEATURES[j], r, d))
if strong:
    for f1, f2, r, d in sorted(strong, key=lambda x: -abs(x[2])):
        print(f"  {f1:15s} <-> {f2:15s}  =  {d}{abs(r):.4f}")
else:
    print("  (无)")

print(f"\n4. 中等相关对 (0.5 <= |r| < 0.7):")
mid = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = corr.iloc[i, j]
        if 0.5 <= abs(r) < 0.7:
            d = '+' if r > 0 else '-'
            mid.append((FEATURES[i], FEATURES[j], r, d))
if mid:
    for f1, f2, r, d in sorted(mid, key=lambda x: -abs(x[2])):
        print(f"  {f1:15s} <-> {f2:15s}  =  {d}{abs(r):.4f}")

# ======================== 5. 热力图 ========================
print("\n5. 绘制热力图...")
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
            vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5,
            mask=mask, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('KMeans 特征相关性矩阵', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PIC_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: pics/correlation_heatmap.png")

# ======================== 6. 特征描述统计 ========================
print("\n6. 特征统计描述:")
stats = df[FEATURES].describe().round(3)
print(f"\n{stats.to_string()}")
stats.to_csv(os.path.join(RESULT_DIR, 'feature_stats.csv'), encoding='utf-8-sig')

print(f"\n完成！")
