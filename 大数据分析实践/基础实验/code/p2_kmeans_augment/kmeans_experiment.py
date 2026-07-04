"""
实验：KMeans 聚类 — NBA 球员技术风格分组（改进版）
功能：使用 KMeans 对球员进行无监督聚类，与真实位置对比
预处理：per-36 归一化、TOT 去重、MP/G 过滤
特征：PTS_per36, TRB_per36, AST_per36, STL_per36, BLK_per36,
       TOV_per36, PF_per36, 3PAr, FTr（9 维）
策略：K=最优（轮廓系数）+ K=5（设计参考）双方案
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# ==================== 路径设置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'NBA_Season_Stats.csv')
PIC_DIR = os.path.join(BASE_DIR, 'pics')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 常数定义
POS_ORDER = ['C', 'PF', 'SF', 'SG', 'PG']
FEATURES = ['PTS_per36', 'TRB_per36', 'AST_per36', 'STL_per36',
            'BLK_per36', 'TOV_per36', 'PF_per36', '3PAr', 'FTr']
N_FEATURES = len(FEATURES)
POS_COLORS = {'C': '#E41A1C', 'PF': '#377EB8', 'SF': '#4DAF4A', 'SG': '#FF7F00', 'PG': '#984EA3'}

# 原始字段到 per-36 字段的映射
PER36_MAP = {
    'PTS_per36': 'PTS', 'TRB_per36': 'TRB', 'AST_per36': 'AST',
    'STL_per36': 'STL', 'BLK_per36': 'BLK',
    'TOV_per36': 'TOV', 'PF_per36': 'PF',
}


# ==================== 1. 数据加载与预处理 ====================

def load_and_clean_data():
    """读取数据，per-36 归一化，TOT 去重，过滤样本"""
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # Per-36 归一化
    for new_f, raw_f in PER36_MAP.items():
        df[new_f] = df[raw_f] / df['MP'].replace(0, np.nan) * 36

    # 3PAr 与 FTr
    df['3PAr'] = df['3PA'] / df['FGA'].replace(0, np.nan)
    df['FTr'] = df['FTA'] / df['FGA'].replace(0, np.nan)

    # 缺失值填充
    for col in ['3PAr', 'FTr']:
        df[col] = df[col].fillna(0)

    # TOT 去重：有 TOT 的球员只保留 TOT 行
    has_tot = df.groupby(['Year', 'Player'])['Tm'].transform(lambda x: (x == 'TOT').any())
    df = df[~(has_tot & (df['Tm'] != 'TOT'))].copy()

    # 行过滤
    df = df[df['MP'] >= 200]
    df = df[df['G'] >= 10]
    df = df.reset_index(drop=True)

    print(f"预处理后: {len(df)} 条")
    for p in POS_ORDER:
        cnt = (df['Pos'] == p).sum()
        print(f"  {p}: {cnt} ({cnt/len(df)*100:.1f}%)")

    return df


# ==================== 2. 特征准备与标准化 ====================

def prepare_features(df):
    """构建特征矩阵"""
    print("\n" + "=" * 60)
    print("2. 特征准备（9 维）")
    print("=" * 60)

    X = df[FEATURES].values
    print(f"特征矩阵: {X.shape}")
    print(f"特征: {FEATURES}")
    return X


def standardize_features(X):
    """Z-score 标准化"""
    print("\n" + "=" * 60)
    print("3. Z-score 标准化")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stats_df = pd.DataFrame({
        '特征': FEATURES,
        '均值': [f"{X[:, i].mean():.3f}" for i in range(N_FEATURES)],
        '标准差': [f"{X[:, i].std():.3f}" for i in range(N_FEATURES)],
    })
    print(stats_df.to_string(index=False))
    print("标准化完成")

    return X_scaled, scaler


# ==================== 3. K 值选择 ====================

def evaluate_k_values(X_scaled):
    """遍历 K=2~10，用轮廓系数选择最优 K"""
    print("\n" + "=" * 60)
    print("4. K 值选择评估（K=2~10）")
    print("=" * 60)

    K_RANGE = range(2, 11)
    inertias = []
    sil_scores = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        print(f"  K={k:2d}  |  Inertia={km.inertia_:.0f}  |  Silhouette={sil:.4f}")

    best_k = K_RANGE[np.argmax(sil_scores)]
    print(f"\n>> 最优 K (轮廓系数): {best_k}")

    return best_k, inertias, sil_scores, K_RANGE


def save_k_selection(best_k, inertias, sil_scores, K_RANGE):
    """保存 K 值选择数据"""
    pd.DataFrame({
        'K': list(K_RANGE), 'Inertia': inertias, 'Silhouette': sil_scores
    }).to_csv(os.path.join(RESULT_DIR, 'k_selection.csv'), index=False, encoding='utf-8-sig')


# ==================== 4. PCA 降维 ====================

def apply_pca(X_scaled):
    """PCA 降维到 2 维用于可视化"""
    print("\n" + "=" * 60)
    print("5. PCA 降维（可视化用）")
    print("=" * 60)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    print(f"PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%}, 累计={var_ratio.sum():.1%}")

    # 保存载荷
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=FEATURES)
    loadings.to_csv(os.path.join(RESULT_DIR, 'pca_loadings.csv'), encoding='utf-8-sig')

    return X_pca, pca, var_ratio


# ==================== 5. 聚类运行 ====================

def run_kmeans(X_scaled, k):
    """运行单次 KMeans 聚类"""
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return km, labels, sil


def run_clustering_experiments(X_scaled, best_k):
    """运行 K=best_k 和 K=5 两组聚类"""
    K_VALUES = [best_k, 5]
    results = {}

    for idx, k in enumerate(K_VALUES):
        print(f"\n" + "=" * 60)
        print(f"6.{idx+1}  K={k} 聚类")
        print("=" * 60)

        km, labels, sil = run_kmeans(X_scaled, k)
        results[k] = {'model': km, 'labels': labels, 'silhouette': sil}

        print(f"Silhouette: {sil:.4f}")
        print(f"\n各簇样本量:")
        for c, cnt in pd.Series(labels).value_counts().sort_index().items():
            print(f"  簇 {c}: {cnt} ({cnt/len(labels)*100:.1f}%)")

    return results, K_VALUES


def print_and_save_cross_tabs(df, results, K_VALUES):
    """打印并保存簇×位置交叉表、簇中心、ARI/NMI"""
    for k in K_VALUES:
        labels = results[k]['labels']
        df[f'C{k}'] = labels

        # 交叉表
        cross = pd.crosstab(df[f'C{k}'], df['Pos'])
        cross_pct = pd.crosstab(df[f'C{k}'], df['Pos'], normalize='index').round(3) * 100

        print(f"\n簇 × 位置 交叉表 (K={k}):")
        print(cross.to_string())
        print(f"\n簇 × 位置 百分比 (K={k}):")
        print(cross_pct.to_string())

        # ARI / NMI
        ari = adjusted_rand_score(df['Pos'], labels)
        nmi = normalized_mutual_info_score(df['Pos'], labels)
        print(f"\nARI: {ari:.4f}  |  NMI: {nmi:.4f}")

        # 保存交叉表
        cross.to_csv(os.path.join(RESULT_DIR, f'cross_table_K{k}.csv'), encoding='utf-8-sig')
        cross_pct.to_csv(os.path.join(RESULT_DIR, f'cross_table_pct_K{k}.csv'), encoding='utf-8-sig')

        # 簇中心（原始尺度）
        centers = pd.DataFrame(
            results[k]['model'].cluster_centers_,
            columns=FEATURES
        )
        centers_raw = pd.DataFrame(
            results[k]['model'].cluster_centers_, columns=FEATURES
        )
        print(f"\n簇中心 (标准化尺度, K={k}):")
        print(centers.round(3).to_string())
        centers.to_csv(os.path.join(RESULT_DIR, f'cluster_centers_K{k}.csv'), encoding='utf-8-sig')


# ==================== 6. 绘图函数 ====================

def draw_k_selection(best_k, inertias, sil_scores, K_RANGE):
    """画 K 值选择图：肘部法 + 轮廓系数"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(K_RANGE), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最优 K={best_k}')
    axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
    axes[0].set_title('肘部法'); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].set_xticks(list(K_RANGE))

    axes[1].plot(list(K_RANGE), sil_scores, 'rs-', linewidth=2, markersize=8)
    axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最优 K={best_k}')
    axes[1].axvline(x=5, color='green', linestyle=':', alpha=0.7, label='K=5（设计参考）')
    axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('轮廓系数'); axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[1].set_xticks(list(K_RANGE))

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'k_selection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pics/k_selection.png")


def draw_pca_clusters(X_pca, df, K_VALUES, results, var_ratio):
    """画 PCA 散点图：聚类结果 vs 真实位置"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for row, k in enumerate(K_VALUES):
        labels = results[k]['labels']
        colors_cluster = plt.cm.Set1

        # 左列：按聚类着色
        ax = axes[row, 0]
        for c in range(k):
            mask = labels == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors_cluster(c)], label=f'簇 {c}',
                       s=12, alpha=0.5, edgecolors='none')
        ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})')
        ax.set_title(f'KMeans 聚类结果 (K={k}, Sil={results[k]["silhouette"]:.3f})')
        ax.legend(fontsize=9, markerscale=2)
        ax.grid(True, alpha=0.2)

        # 右列：按真实位置着色
        ax = axes[row, 1]
        for pos in POS_ORDER:
            mask = df['Pos'] == pos
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=POS_COLORS[pos], label=pos, s=12, alpha=0.5, edgecolors='none')
        ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})')
        ax.set_title(f'真实位置分布 (K={k} 参考)')
        ax.legend(fontsize=11, markerscale=2)
        ax.grid(True, alpha=0.2)

    plt.suptitle('KMeans 聚类 vs 真实位置 (PCA 降维)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'pca_clusters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pics/pca_clusters.png")


def draw_radar(results, k):
    """画雷达图"""
    angles = np.linspace(0, 2 * np.pi, N_FEATURES, endpoint=False).tolist()
    angles += angles[:1]

    centers_scaled = results[k]['model'].cluster_centers_
    cmin = centers_scaled.min(axis=0)
    cmax = centers_scaled.max(axis=0)
    centers_norm = (centers_scaled - cmin) / (cmax - cmin + 1e-10)
    colors_cluster = plt.cm.Set1

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for c in range(k):
        values = centers_norm[c].tolist() + [centers_norm[c][0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'簇 {c}', color=colors_cluster(c))
        ax.fill(angles, values, alpha=0.05, color=colors_cluster(c))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURES, fontsize=10)
    ax.set_title(f'KMeans 簇中心雷达图 (K={k}, 归一化)', fontsize=14, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'radar_K{k}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/radar_K{k}.png")


def draw_feature_comparison(scaler, results, k):
    """画各簇特征均值对比柱状图"""
    centers_raw = pd.DataFrame(
        scaler.inverse_transform(results[k]['model'].cluster_centers_),
        columns=FEATURES
    ).round(2)
    colors_cluster = plt.cm.Set1

    n_cols = 3
    n_rows = int(np.ceil(N_FEATURES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        bars = ax.bar(range(k), centers_raw[feat],
                      color=[colors_cluster(c) for c in range(k)],
                      edgecolor='black', linewidth=0.5)
        ax.set_title(feat, fontweight='bold')
        ax.set_xticks(range(k))
        ax.set_xticklabels([f'簇{c}' for c in range(k)])
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, centers_raw[feat]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'K={k} 各簇特征均值对比', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'feature_comparison_K{k}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/feature_comparison_K{k}.png")


def draw_pie_comparison(df, K_VALUES):
    """画簇大小饼图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_cluster = plt.cm.Set1

    for idx, k in enumerate(K_VALUES):
        ax = axes[idx]
        sizes = df[f'C{k}'].value_counts().sort_index()
        wedges, texts, autotexts = ax.pie(
            sizes.values,
            labels=[f'簇{c}\n({n}条)' for c, n in sizes.items()],
            colors=[colors_cluster(c) for c in sizes.index],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9}
        )
        ax.set_title(f'聚类分布 (K={k})', fontsize=13)

    plt.suptitle('KMeans 聚类样本分布对比', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'cluster_pie.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pics/cluster_pie.png")


# ==================== 7. 保存结果 ====================

def save_results(df, results, K_VALUES):
    """保存评估汇总和全量数据"""
    # 评估汇总
    summary = []
    for k in K_VALUES:
        ari = adjusted_rand_score(df['Pos'], results[k]['labels'])
        nmi = normalized_mutual_info_score(df['Pos'], results[k]['labels'])
        summary.append({
            'K': k,
            'Silhouette': round(results[k]['silhouette'], 4),
            'ARI': round(ari, 4),
            'NMI': round(nmi, 4),
        })
        print(f"K={k:2d}  |  Silhouette={results[k]['silhouette']:.4f}  |  ARI={ari:.4f}  |  NMI={nmi:.4f}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULT_DIR, 'evaluation_summary.csv'), index=False, encoding='utf-8-sig')

    # 全量数据（含聚类标签）
    output_cols = ['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'MP'] + FEATURES + [f'C{k}' for k in K_VALUES]
    df[output_cols].to_csv(os.path.join(RESULT_DIR, 'clustered_data.csv'), index=False, encoding='utf-8-sig')


# ==================== 8. 主流程 ====================

def main():
    print("=" * 60)
    print("KMeans 聚类 — NBA 球员技术风格分组（改进版）")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_clean_data()

    # 2. 特征准备
    X = prepare_features(df)

    # 3. 标准化
    X_scaled, scaler = standardize_features(X)

    # 4. K 值选择
    best_k, inertias, sil_scores, K_RANGE = evaluate_k_values(X_scaled)
    save_k_selection(best_k, inertias, sil_scores, K_RANGE)

    # 5. PCA 降维
    X_pca, pca, var_ratio = apply_pca(X_scaled)

    # 6. 聚类实验
    results, K_VALUES = run_clustering_experiments(X_scaled, best_k)
    print_and_save_cross_tabs(df, results, K_VALUES)

    # 7. 画图
    print("\n" + "=" * 60)
    print("7. 可视化")
    print("=" * 60)

    draw_k_selection(best_k, inertias, sil_scores, K_RANGE)
    draw_pca_clusters(X_pca, df, K_VALUES, results, var_ratio)

    for k in K_VALUES:
        draw_radar(results, k)
        draw_feature_comparison(scaler, results, k)

    draw_pie_comparison(df, K_VALUES)

    # 8. 保存结果
    print("\n" + "=" * 60)
    print("8. 评估汇总")
    print("=" * 60)
    save_results(df, results, K_VALUES)


if __name__ == '__main__':
    main()
