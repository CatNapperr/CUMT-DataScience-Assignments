"""
实验：KMeans 聚类 — NBA 球员技术风格分组（原始版）
功能：使用 KMeans 对球员进行无监督聚类，与真实场上位置对比
预处理：场均统计、缺失值填充、G>=20 过滤、删除 TOT 行
特征：PPG, APG, RPG, SPG, BPG, FG%, 3P%, FT%（8 维）
策略：K=最优（轮廓系数）+ K=5（与位置数对应）双方案对比
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
PIC_DIR = os.path.join(BASE_DIR, 'pic')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 常数定义
POS_ORDER = ['C', 'PF', 'SF', 'SG', 'PG']
FEATURE_COLS = ['PPG', 'APG', 'RPG', 'SPG', 'BPG', 'FG%', '3P%', 'FT%']
POS_COLORS = {'C': '#E41A1C', 'PF': '#377EB8', 'SF': '#4DAF4A', 'SG': '#FF7F00', 'PG': '#984EA3'}


# ==================== 1. 数据加载与预处理 ====================

def load_and_clean_data():
    """读取数据，计算场均统计，填充缺失值，过滤样本"""
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # 场均统计
    df['PPG'] = df['PTS'] / df['G']
    df['APG'] = df['AST'] / df['G']
    df['RPG'] = df['TRB'] / df['G']
    df['SPG'] = df['STL'] / df['G']
    df['BPG'] = df['BLK'] / df['G']
    df['FPG'] = df['PF'] / df['G']
    df['TOPG'] = df['TOV'] / df['G']
    print("场均字段: PPG, APG, RPG, SPG, BPG, FPG, TOPG")

    # 缺失值填充
    for col in ['FG%', '3P%', 'FT%']:
        df[col] = df[col].fillna(0)

    # 过滤
    n_before = len(df)
    df = df[df['G'] >= 20]
    print(f"剔除 G<20: {n_before - len(df)} 条")
    df = df[df['Tm'] != 'TOT']
    df = df.reset_index(drop=True)
    print(f"最终有效样本: {len(df)} 条")
    for pos in POS_ORDER:
        cnt = (df['Pos'] == pos).sum()
        print(f"  {pos}: {cnt} ({cnt/len(df)*100:.1f}%)")

    return df


# ==================== 2. 特征准备与标准化 ====================

def prepare_features(df):
    """构建特征矩阵"""
    print("\n" + "=" * 60)
    print("2. 特征准备（8 维）")
    print("=" * 60)

    X = df[FEATURE_COLS].values
    print(f"特征矩阵: {X.shape}")
    print(f"特征: {FEATURE_COLS}")
    return X


def standardize_features(X):
    """Z-score 标准化"""
    print("\n" + "=" * 60)
    print("3. Z-score 标准化")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stats_df = pd.DataFrame({
        '特征': FEATURE_COLS,
        '均值': [f"{X[:, i].mean():.3f}" for i in range(len(FEATURE_COLS))],
        '标准差': [f"{X[:, i].std():.3f}" for i in range(len(FEATURE_COLS))],
    })
    print(stats_df.to_string(index=False))
    print("标准化后: 均值≈0, 标准差≈1")

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
    print(f"PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}, 累计={var_ratio.sum():.2%}")

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=FEATURE_COLS)
    print("\nPC 载荷:")
    print(loadings.round(3).to_string())

    return X_pca, pca


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

    for k in K_VALUES:
        print(f"\n" + "=" * 60)
        print(f"6.{K_VALUES.index(k)+1}  K={k} 聚类")
        print("=" * 60)

        km, labels, sil = run_kmeans(X_scaled, k)
        results[k] = {'model': km, 'labels': labels, 'silhouette': sil}

        print(f"Silhouette: {sil:.4f}")
        print(f"\n各簇样本量:")
        for c, cnt in pd.Series(labels).value_counts().sort_index().items():
            print(f"  簇 {c}: {cnt} ({cnt/len(labels)*100:.1f}%)")

    return results, K_VALUES


def print_cross_tabs(df, results, K_VALUES):
    """打印并保存簇×位置交叉表"""
    for k in K_VALUES:
        labels = results[k]['labels']
        df[f'Cluster_{k}'] = labels

        # 交叉表
        cross = pd.crosstab(df[f'Cluster_{k}'], df['Pos'])
        cross_pct = pd.crosstab(df[f'Cluster_{k}'], df['Pos'], normalize='index').round(3) * 100

        print(f"\n簇 × 位置 交叉表 (K={k}):")
        print(cross.to_string())
        print(f"\n簇 × 位置 百分比 (K={k}):")
        print(cross_pct.to_string())

        # ARI / NMI
        ari = adjusted_rand_score(df['Pos'], labels)
        nmi = normalized_mutual_info_score(df['Pos'], labels)
        print(f"\nARI: {ari:.4f}  NMI: {nmi:.4f}")

        cross.to_csv(os.path.join(RESULT_DIR, f'cross_table_K{k}.csv'), encoding='utf-8-sig')
        cross_pct.to_csv(os.path.join(RESULT_DIR, f'cross_table_pct_K{k}.csv'), encoding='utf-8-sig')

        # 簇中心（原始尺度）
        centers_raw = df.groupby(f'Cluster_{k}')[FEATURE_COLS].mean().round(2)
        centers_raw.to_csv(os.path.join(RESULT_DIR, f'cluster_centers_K{k}.csv'), encoding='utf-8-sig')
        print(f"\n簇中心特征均值 (K={k}):")
        print(centers_raw.to_string())

        # 簇→位置映射
        print(f"\n簇 → 位置映射 (K={k}):")
        for c in range(k):
            cluster_data = df[df[f'Cluster_{k}'] == c]
            pos_dist = cluster_data['Pos'].value_counts()
            top_pos = pos_dist.index[0]
            top_pct = pos_dist.iloc[0] / len(cluster_data) * 100
            top3 = ', '.join([f'{p}({v/len(cluster_data)*100:.0f}%)' for p, v in pos_dist.head(3).items()])
            print(f"  簇 {c}: 主要 = {top_pos} ({top_pct:.1f}%), 前3: {top3}")


# ==================== 6. 绘图函数 ====================

def draw_k_selection(best_k, inertias, sil_scores, K_RANGE):
    """画 K 值选择图：肘部法 + 轮廓系数"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 肘部法
    axes[0].plot(list(K_RANGE), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最优 K={best_k}')
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('肘部法', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xticks(list(K_RANGE))

    # 轮廓系数
    axes[1].plot(list(K_RANGE), sil_scores, 'rs-', linewidth=2, markersize=8)
    axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最优 K={best_k}')
    axes[1].axvline(x=5, color='green', linestyle=':', alpha=0.7, label='K=5（设计参考）')
    axes[1].set_xlabel('K', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('轮廓系数', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xticks(list(K_RANGE))

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'k_selection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/k_selection.png")


def draw_pca_clusters(X_pca, df, K_VALUES, results, pca):
    """画 PCA 散点图：聚类结果 vs 真实位置"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for row, k in enumerate(K_VALUES):
        labels = results[k]['labels']

        # 左列：按聚类着色
        ax = axes[row, 0]
        colors_cluster = plt.cm.Set1(np.linspace(0, 1, k))
        for c in range(k):
            mask = labels == c
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors_cluster[c]], label=f'簇 {c}',
                       s=12, alpha=0.5, edgecolors='none')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        ax.set_title(f'KMeans 聚类结果 (K={k})', fontsize=13)
        ax.legend(fontsize=9, markerscale=2)
        ax.grid(True, alpha=0.2)

        # 右列：按真实位置着色
        ax = axes[row, 1]
        for pos in POS_ORDER:
            mask = df['Pos'] == pos
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[POS_COLORS[pos]], label=pos, s=12, alpha=0.5, edgecolors='none')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        ax.set_title(f'真实位置分布 (K={k} 参考)', fontsize=13)
        ax.legend(fontsize=11, markerscale=2)
        ax.grid(True, alpha=0.2)

    plt.suptitle('KMeans 聚类 vs 真实位置 (PCA 降维)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'pca_clusters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/pca_clusters.png")


def draw_radar(centers_scaled, k, title, filename):
    """画雷达图"""
    n_features = len(FEATURE_COLS)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    centers_norm = (centers_scaled - centers_scaled.min(axis=0)) / (centers_scaled.max(axis=0) - centers_scaled.min(axis=0) + 1e-10)
    colors = plt.cm.Set1(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c in range(k):
        values = centers_norm[c].tolist() + [centers_norm[c][0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'簇 {c}', color=colors[c])
        ax.fill(angles, values, alpha=0.08, color=colors[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURE_COLS, fontsize=11)
    ax.set_title(title, fontsize=14, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pic/{filename}")


def draw_feature_comparison(centers_raw, k, title, filename):
    """画各簇特征均值对比柱状图"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    colors = plt.cm.Set1(np.linspace(0, 1, k))

    for i, feat in enumerate(FEATURE_COLS):
        ax = axes[i]
        bars = ax.bar(range(k), centers_raw[feat], color=[colors[c] for c in range(k)],
                      edgecolor='black', linewidth=0.5)
        ax.set_title(feat, fontsize=13, fontweight='bold')
        ax.set_xticks(range(k))
        ax.set_xticklabels([f'簇{c}' for c in range(k)], fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, centers_raw[feat]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pic/{filename}")


def draw_pie_comparison(df, K_VALUES):
    """画簇大小饼图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, k in enumerate(K_VALUES):
        ax = axes[idx]
        sizes = df[f'Cluster_{k}'].value_counts().sort_index()
        colors_k = plt.cm.Set1(np.linspace(0, 1, k))
        wedges, texts, autotexts = ax.pie(
            sizes.values, labels=[f'簇{c}\n({n}条)' for c, n in sizes.items()],
            colors=[colors_k[c] for c in sizes.index],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9}
        )
        ax.set_title(f'聚类分布 (K={k})', fontsize=13)
    plt.suptitle('KMeans 聚类样本分布对比', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'cluster_pie_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/cluster_pie_comparison.png")


# ==================== 8. 保存结果 ====================

def save_results(df, results, K_VALUES, best_k):
    """保存清洗后数据"""
    clean_data = df[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G'] + FEATURE_COLS].copy()
    clean_data.to_csv(os.path.join(RESULT_DIR, 'clean_data.csv'), index=False, encoding='utf-8-sig')
    print("清洗后数据已保存: result/clean_data.csv")


# ==================== 9. 主流程 ====================

def main():
    print("=" * 60)
    print("KMeans 聚类 — NBA 球员技术风格分组（原始版）")
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
    X_pca, pca = apply_pca(X_scaled)

    # 6. 聚类实验
    results, K_VALUES = run_clustering_experiments(X_scaled, best_k)
    print_cross_tabs(df, results, K_VALUES)

    # 7. 画图
    print("\n" + "=" * 60)
    print("8. 可视化")
    print("=" * 60)

    draw_k_selection(best_k, inertias, sil_scores, K_RANGE)
    draw_pca_clusters(X_pca, df, K_VALUES, results, pca)

    for k in [5, best_k]:
        centers_scaled = results[k]['model'].cluster_centers_
        draw_radar(centers_scaled, k, f'KMeans 簇中心雷达图 (K={k}, 标准化)', f'radar_K{k}.png')

        centers_raw_df = pd.DataFrame(
            scaler.inverse_transform(centers_scaled), columns=FEATURE_COLS
        ).round(2)
        draw_feature_comparison(centers_raw_df, k, f'K={k} 各簇特征均值对比', f'feature_comparison_K{k}.png')

    draw_pie_comparison(df, K_VALUES)

    # 8. 保存结果
    save_results(df, results, K_VALUES, best_k)

    # 9. 最终汇总
    print("=" * 60)
    print("KMeans 聚类实验完成！")
    print("=" * 60)

    print(f"""
实验结果对比:
────────────────────────────────────────────────
方案        K       Silhouette    ARI       NMI
────────────────────────────────────────────────
自动最优    {best_k}       {results[best_k]['silhouette']:.4f}      {adjusted_rand_score(df['Pos'], results[best_k]['labels']):.4f}      {normalized_mutual_info_score(df['Pos'], results[best_k]['labels']):.4f}
设计参考    5       {results[5]['silhouette']:.4f}      {adjusted_rand_score(df['Pos'], results[5]['labels']):.4f}      {normalized_mutual_info_score(df['Pos'], results[5]['labels']):.4f}
────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
