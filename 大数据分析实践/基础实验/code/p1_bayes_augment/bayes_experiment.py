"""
实验：朴素贝叶斯分类 — NBA 球员位置预测（改进版）
改进点：Per-36 分钟归一化、TOT 去重、新增高阶特征
同时运行 5 位置分类和 3 超类分类两组实验
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ==================== 路径设置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'NBA_Season_Stats.csv')
PIC_DIR = os.path.join(BASE_DIR, 'pics')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 位置定义
POS_ORDER = ['C', 'PF', 'SF', 'SG', 'PG']
SUPER_ORDER = ['Big', 'Wing', 'PG']
POS_TO_SUPER = {'C': 'Big', 'PF': 'Big', 'PG': 'PG', 'SF': 'Wing', 'SG': 'Wing'}

# 特征列表
FEATURES = [
    'PTS_per36', 'TRB_per36', 'AST_per36', 'STL_per36', 'BLK_per36',
    '3P%', 'FT%', 'eFG%',
    '3PA_per36', 'FTA_per36', 'TOV_per36', 'PF_per36',
]


# ==================== 1. 数据预处理 ====================

def load_and_clean_data():
    """
    读取数据，做 per-36 归一化，TOT 去重，过滤样本
    """
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # 计算有效命中率
    df['eFG%'] = df['eFG%'].fillna((df['FG'] + 0.5 * df['3P']) / df['FGA'])

    # 将累计统计换算成每 36 分钟的数据
    per36_cols = {
        'PTS_per36': 'PTS', 'TRB_per36': 'TRB', 'AST_per36': 'AST',
        'STL_per36': 'STL', 'BLK_per36': 'BLK', 'PF_per36': 'PF',
        'TOV_per36': 'TOV', '3PA_per36': '3PA', 'FTA_per36': 'FTA',
    }
    for new_col, raw_col in per36_cols.items():
        df[new_col] = df[raw_col] / df['MP'].replace(0, np.nan) * 36

    # 命中率缺失补 0
    for col in ['FG%', '3P%', 'FT%', '2P%', 'eFG%']:
        df[col] = df[col].fillna(0)

    # TOT 去重：如果一个球员赛季有 TOT 行，只保留 TOT，删除各球队的行
    has_tot = df.groupby(['Year', 'Player'])['Tm'].transform(lambda x: (x == 'TOT').any())
    df = df[~(has_tot & (df['Tm'] != 'TOT'))].copy()

    # 过滤：至少打 200 分钟，至少出场 10 场
    df = df[df['MP'] >= 200]
    df = df[df['G'] >= 10]
    df = df.reset_index(drop=True)

    print(f"预处理后: {len(df)} 条")
    return df


# ==================== 2. 准备特征和标签 ====================

def prepare_labels(df):
    """
    构造 5 位置和 3 超类的标签
    """
    print("\n" + "=" * 60)
    print("2. 准备特征和标签")
    print("=" * 60)

    X = df[FEATURES].values

    # 5 位置标签
    le5 = LabelEncoder()
    le5.fit(POS_ORDER)
    y5 = le5.transform(df['Pos'])

    # 3 超类标签
    df['SuperPos'] = df['Pos'].map(POS_TO_SUPER)
    le3 = LabelEncoder()
    le3.fit(SUPER_ORDER)
    y3 = le3.transform(df['SuperPos'])

    print(f"特征: {len(FEATURES)} 维")
    print(f"5 类分布: {dict(zip(POS_ORDER, np.bincount(y5)))}")
    print(f"3 类分布: {dict(zip(SUPER_ORDER, np.bincount(y3)))}")

    return X, y5, y3, le5, le3


# ==================== 3. 划分数据集 ====================

def split_data(X, y5, y3):
    """
    按 7:3 分层采样划分，保证 5 类和 3 类使用同一份训练/测试集
    """
    print("\n" + "=" * 60)
    print("3. 数据划分（7:3 分层采样）")
    print("=" * 60)

    X_train, X_test, y5_train, y5_test, y3_train, y3_test = train_test_split(
        X, y5, y3, test_size=0.3, random_state=42, stratify=y5
    )
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    return X_train, X_test, y5_train, y5_test, y3_train, y3_test


# ==================== 4. 训练和评估 ====================

def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, exp_name):
    """
    训练 GaussianNB 并输出分类结果
    """
    print(f"\n--- {exp_name} ---")

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")

    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'report': report,
    }


def evaluate_5to3_mapping(y5_pred, le5, le3, y3_test):
    """
    将 5 类预测结果映射到 3 超类，计算映射后的准确率
    """
    y5_super = np.array([POS_TO_SUPER[le5.inverse_transform([p])[0]] for p in y5_pred])
    y5_super_enc = le3.transform(y5_super)
    acc = accuracy_score(y3_test, y5_super_enc)
    cm = confusion_matrix(y3_test, y5_super_enc)
    macro_f1 = f1_score(y3_test, y5_super_enc, average='macro')
    return acc, cm, macro_f1, y5_super_enc


# ==================== 5. 画图 ====================

def draw_cm_5class(cm, acc):
    """画 5 分类混淆矩阵"""
    fig, ax = plt.subplots(figsize=(9, 8))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=POS_ORDER, yticklabels=POS_ORDER, ax=ax, vmin=0, vmax=100)
    ax.set_xlabel('预测位置')
    ax.set_ylabel('真实位置')
    ax.set_title(f'5 分类 — 混淆矩阵（%）, Acc={acc:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'cm_5class.png'), dpi=150, bbox_inches='tight')
    plt.close()


def draw_cm_3class(cm_3class, cm_5to3, acc_3class, acc_5to3):
    """画 3 超类混淆矩阵对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm_data, title in zip(axes,
                                  [cm_3class, cm_5to3],
                                  [f'3 超类 — 直接分类\nAcc={acc_3class:.3f}',
                                   f'5 类 -> 3 类映射\nAcc={acc_5to3:.3f}']):
        cm_pct = cm_data.astype(float) / cm_data.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=SUPER_ORDER, yticklabels=SUPER_ORDER, ax=ax, vmin=0, vmax=100)
        ax.set_xlabel('预测超类')
        ax.set_ylabel('真实超类')
        ax.set_title(title)
    plt.suptitle('3 超类分类 — 混淆矩阵对比（%）', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'cm_3class.png'), dpi=150, bbox_inches='tight')
    plt.close()


def draw_comparison_bar(res5, res3):
    """画 5 类 vs 3 类指标对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    vals_5 = [res5['accuracy'], res5['macro_f1'], res5['weighted_f1']]
    vals_3 = [res3['accuracy'], res3['macro_f1'], res3['weighted_f1']]
    x = np.arange(len(metrics))
    ax.bar(x - 0.15, vals_5, 0.3, label='5 位置分类', color='#E67E22', edgecolor='black')
    ax.bar(x + 0.15, vals_3, 0.3, label='3 超类分类', color='#4A90D9', edgecolor='black')
    for i, (v5, v3) in enumerate(zip(vals_5, vals_3)):
        ax.text(i - 0.15, v5 + 0.005, f'{v5:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + 0.15, v3 + 0.005, f'{v3:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('分数')
    ax.set_title('5 位置 vs 3 超类 — 分类性能对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'comparison_5vs3.png'), dpi=150, bbox_inches='tight')
    plt.close()


def draw_metrics_3class(y_test, y_pred):
    """画 3 超类各超类的 P/R/F1"""
    report = classification_report(y_test, y_pred, target_names=SUPER_ORDER, output_dict=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SUPER_ORDER))
    for i, metric in enumerate(['精确率', '召回率', 'F1']):
        key = {'精确率': 'precision', '召回率': 'recall', 'F1': 'f1-score'}[metric]
        vals = [report[cls][key] for cls in SUPER_ORDER]
        ax.bar(x + (i - 1) * 0.25, vals, 0.25, label=metric, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(SUPER_ORDER)
    ax.set_ylabel('分数')
    ax.set_title('3 超类 — 各超类分类指标')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'metrics_3class.png'), dpi=150, bbox_inches='tight')
    plt.close()


def draw_metrics_5class(res5):
    """画 5 类各位置的 P/R/F1"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(POS_ORDER))
    for i, metric in enumerate(['精确率', '召回率', 'F1']):
        key = {'精确率': 'precision', '召回率': 'recall', 'F1': 'f1-score'}[metric]
        vals = [res5['report'][cls][key] for cls in POS_ORDER]
        ax.bar(x + (i - 1) * 0.25, vals, 0.25, label=metric, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(POS_ORDER)
    ax.set_ylabel('分数')
    ax.set_title('5 分类 — 各位置分类指标')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'metrics_5class.png'), dpi=150, bbox_inches='tight')
    plt.close()


def draw_feature_distribution(df):
    """画特征分布箱线图（5 类和 3 类两行）"""
    plot_features = ['PTS_per36', 'TRB_per36', 'AST_per36', 'BLK_per36', '3PA_per36', '3P%']
    colors_5 = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3']
    colors_3 = ['#E41A1C', '#4DAF4A', '#984EA3']

    fig, axes = plt.subplots(2, len(plot_features), figsize=(22, 12))
    for j, feat in enumerate(plot_features):
        # 第一行：5 类
        ax = axes[0, j]
        data = [df[df['Pos'] == p][feat].dropna().values for p in POS_ORDER]
        bp = ax.boxplot(data, tick_labels=POS_ORDER, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_5):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.set_title(f'{feat}（5 类）', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 第二行：3 类
        ax = axes[1, j]
        data = [df[df['SuperPos'] == c][feat].dropna().values for c in SUPER_ORDER]
        bp = ax.boxplot(data, tick_labels=SUPER_ORDER, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_3):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.set_title(f'{feat}（3 超类）', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('特征分布箱线图 — 5 分类 vs 3 超类', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'feature_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 6. 保存结果 ====================

def save_results(res5, res3, acc_map, cm_map, y3_test, y5_pred_super_enc, df):
    """保存所有 CSV 结果"""
    # 分类报告
    pd.DataFrame(res5['report']).transpose().to_csv(
        os.path.join(RESULT_DIR, 'class_report_5class.csv'), encoding='utf-8-sig')
    pd.DataFrame(res3['report']).transpose().to_csv(
        os.path.join(RESULT_DIR, 'class_report_3class.csv'), encoding='utf-8-sig')

    # 混淆矩阵
    pd.DataFrame(res5['confusion_matrix'], index=POS_ORDER, columns=POS_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'cm_5class.csv'), encoding='utf-8-sig')
    pd.DataFrame(res3['confusion_matrix'], index=SUPER_ORDER, columns=SUPER_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'cm_3class.csv'), encoding='utf-8-sig')
    pd.DataFrame(cm_map, index=SUPER_ORDER, columns=SUPER_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'cm_5_to_3.csv'), encoding='utf-8-sig')

    # 对比表
    pd.DataFrame({
        'metric': ['Accuracy', 'Macro_F1', 'Weighted_F1'],
        '5class': [res5['accuracy'], res5['macro_f1'], res5['weighted_f1']],
        '5class_mapped_to_3': [acc_map,
                               f1_score(y3_test, y5_pred_super_enc, average='macro'),
                               f1_score(y3_test, y5_pred_super_enc, average='weighted')],
        '3class_direct': [res3['accuracy'], res3['macro_f1'], res3['weighted_f1']],
    }).to_csv(os.path.join(RESULT_DIR, 'comparison_5vs3.csv'), index=False, encoding='utf-8-sig')

    # 特征统计
    df.groupby('Pos')[FEATURES].describe().round(3).to_csv(
        os.path.join(RESULT_DIR, 'position_feature_stats.csv'), encoding='utf-8-sig')

    print("已保存全部结果 CSV")


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("朴素贝叶斯实验 — NBA 球员位置预测（改进版）")
    print(f"特征: {len(FEATURES)} 维 per-36")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_clean_data()

    # 2. 准备标签
    X, y5, y3, le5, le3 = prepare_labels(df)
    X_train, X_test, y5_train, y5_test, y3_train, y3_test = split_data(X, y5, y3)

    # 3. 训练模型
    print("\n" + "=" * 60)
    print("4. 模型训练与评估")
    print("=" * 60)
    res5 = train_and_evaluate(X_train, X_test, y5_train, y5_test, POS_ORDER, "5 分类（12 维）")
    res3 = train_and_evaluate(X_train, X_test, y3_train, y3_test, SUPER_ORDER, "3 超类（12 维）")

    # 4. 5→3 映射
    print("\n" + "=" * 60)
    print("5. 5 类 -> 3 类映射对比")
    print("=" * 60)
    acc_map, cm_map, map_macro_f1, y5_pred_super_enc = evaluate_5to3_mapping(
        res5['y_pred'], le5, le3, y3_test)
    print(f"5 类 -> 3 类映射 Accuracy: {acc_map:.4f}")
    print(f"3 超类直接分类 Accuracy:  {res3['accuracy']:.4f}")
    print(f"提升: {res3['accuracy'] - acc_map:+.4f}")

    # 5. 画图
    print("\n" + "=" * 60)
    print("6. 可视化")
    print("=" * 60)
    draw_cm_5class(res5['confusion_matrix'], res5['accuracy'])
    draw_cm_3class(res3['confusion_matrix'], cm_map, res3['accuracy'], acc_map)
    draw_comparison_bar(res5, res3)
    draw_metrics_3class(y3_test, res3['y_pred'])
    draw_metrics_5class(res5)
    draw_feature_distribution(df)
    print("已保存全部图片到 pics/")

    # 6. 保存结果
    print("\n" + "=" * 60)
    print("7. 保存结果 CSV")
    print("=" * 60)
    save_results(res5, res3, acc_map, cm_map, y3_test, y5_pred_super_enc, df)

    # 7. 打印总结
    print(f"\n{'=' * 60}")
    print(f"实验结果汇总")
    print(f"{'=' * 60}")
    print(f"\n5 分类（12 维）:")
    print(f"  Accuracy:    {res5['accuracy']:.4f}")
    print(f"  Macro F1:    {res5['macro_f1']:.4f}")
    print(f"  Weighted F1: {res5['weighted_f1']:.4f}")
    print(f"\n3 超类 — 直接分类（12 维）:")
    print(f"  Accuracy:    {res3['accuracy']:.4f}")
    print(f"  Macro F1:    {res3['macro_f1']:.4f}")
    print(f"  Weighted F1: {res3['weighted_f1']:.4f}")
    print(f"\n5 类 -> 3 类映射:")
    print(f"  Accuracy:    {acc_map:.4f}")
    print(f"  3 超类相比 5 类映射提升: {res3['accuracy'] - acc_map:+.4f}")
    print(f"\n实验完成！")


if __name__ == '__main__':
    main()
