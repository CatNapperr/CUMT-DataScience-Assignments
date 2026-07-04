"""
实验：朴素贝叶斯分类 — NBA 球员位置预测
功能：使用 GaussianNB 预测球员场上位置，并对比是否加入 Age 特征的效果
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
PIC_DIR = os.path.join(BASE_DIR, 'pic')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 位置顺序固定
POS_ORDER = ['C', 'PF', 'SF', 'SG', 'PG']


# ==================== 1. 数据加载与预处理 ====================

def load_and_clean_data():
    """读取数据，生成场均统计，填充缺失值，过滤样本"""
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # 计算场均数据
    df['PPG'] = df['PTS'] / df['G']
    df['APG'] = df['AST'] / df['G']
    df['RPG'] = df['TRB'] / df['G']
    df['SPG'] = df['STL'] / df['G']
    df['BPG'] = df['BLK'] / df['G']
    df['FPG'] = df['PF'] / df['G']
    df['TOPG'] = df['TOV'] / df['G']

    # 命中率缺失值补 0
    for col in ['FG%', '3P%', 'FT%']:
        df[col] = df[col].fillna(0)

    # 过滤：至少打 20 场，删除 TOT 汇总行
    df = df[df['G'] >= 20]
    df = df[df['Tm'] != 'TOT']
    df = df.reset_index(drop=True)

    print(f"过滤后: {len(df)} 条")
    for pos in POS_ORDER:
        print(f"  {pos}: {(df['Pos'] == pos).sum()}")
    return df


# ==================== 2. 准备特征和标签 ====================

def prepare_data(df):
    """构造特征矩阵和标签，区分无 Age 和含 Age 两组"""
    print("\n" + "=" * 60)
    print("2. 准备特征和标签")
    print("=" * 60)

    core_features = ['PPG', 'APG', 'RPG', 'SPG', 'BPG', 'FG%', '3P%', 'FT%']
    age_feature = ['Age']

    # 编码标签
    le = LabelEncoder()
    le.fit(POS_ORDER)
    y = le.transform(df['Pos'])

    # 无 Age 组
    X_wo = df[core_features].values
    # 含 Age 组
    X_w = df[core_features + age_feature].values

    print(f"无 Age 特征 (8 维): {core_features}")
    print(f"含 Age 特征 (9 维): {core_features + age_feature}")
    print(f"标签分布: {dict(zip(POS_ORDER, np.bincount(y)))}")

    return X_wo, X_w, y, core_features, core_features + age_feature


# ==================== 3. 划分训练集和测试集 ====================

def split_data(X, y):
    """按 7:3 分层采样划分"""
    print("\n" + "=" * 60)
    print("3. 划分训练集/测试集（7:3 分层采样）")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    return X_train, X_test, y_train, y_test


# ==================== 4. 训练和评估 ====================

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_list, group_name):
    """训练 GaussianNB 并输出评估结果"""
    print(f"\n--- {group_name} ---")

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=POS_ORDER, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"准确率: {acc:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=POS_ORDER, zero_division=0)}")

    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'report': report,
        'confusion_matrix': cm,
        'feature_list': feature_list,
    }


# ==================== 5. 画图 ====================

def draw_confusion_matrix_comparison(result_wo, result_w):
    """并排画两个混淆矩阵（无 Age vs 含 Age）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, result, title in zip(axes,
                                 [result_wo, result_w],
                                 [f'NB-A: 无 Age\nAcc={result_wo["accuracy"]:.3f}',
                                  f'NB-B: 含 Age\nAcc={result_w["accuracy"]:.3f}']):
        cm = result['confusion_matrix']
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=POS_ORDER, yticklabels=POS_ORDER,
                    ax=ax, vmin=0, vmax=100)
        ax.set_xlabel('预测位置')
        ax.set_ylabel('真实位置')
        ax.set_title(title)
    plt.suptitle('朴素贝叶斯 混淆矩阵对比（百分比）', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'confusion_matrix_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/confusion_matrix_comparison.png")


def draw_f1_comparison(result_wo, result_w):
    """画各类别 F1 对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(POS_ORDER))
    f1_wo = [result_wo['report'][cls]['f1-score'] for cls in POS_ORDER]
    f1_w = [result_w['report'][cls]['f1-score'] for cls in POS_ORDER]

    ax.bar(x - 0.15, f1_wo, 0.3, label='无 Age', color='#4A90D9', edgecolor='black')
    ax.bar(x + 0.15, f1_w, 0.3, label='含 Age', color='#E67E22', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(POS_ORDER)
    ax.set_ylabel('F1 Score')
    ax.set_title('Age 对照实验 — 各类别 F1 对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/f1_comparison.png")


def draw_other_plots(result, best_name):
    """画最优模型的额外图表"""
    # 混淆矩阵数值版
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=POS_ORDER, yticklabels=POS_ORDER, ax=ax)
    ax.set_xlabel('预测位置')
    ax.set_ylabel('真实位置')
    ax.set_title(f'{best_name} 混淆矩阵（数值）')
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'confusion_matrix_best.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # P/R/F1 汇总
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(POS_ORDER))
    metrics = ['精确率', '召回率', 'F1']
    for i, m in enumerate(metrics):
        vals = [result['report'][cls][{'精确率': 'precision', '召回率': 'recall', 'F1': 'f1-score'}[m]] for cls in POS_ORDER]
        ax.bar(x + (i - 1) * 0.25, vals, 0.25, label=m, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(POS_ORDER)
    ax.set_ylabel('分数')
    ax.set_title(f'{best_name} — 各位置分类指标')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'classification_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 特征分布箱线图
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3']
    features = result['feature_list']
    for i, feat in enumerate(features):
        ax = axes.flatten()[i]
        pos_data = [df[df['Pos'] == pos][feat].dropna().values for pos in POS_ORDER]
        bp = ax.boxplot(pos_data, tick_labels=POS_ORDER, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.set_title(feat, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('各位置在不同特征上的分布对比', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'feature_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/confusion_matrix_best.png, classification_metrics.png, feature_distribution.png")


def draw_overall_comparison(result_wo, result_w):
    """画总体指标对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    vals_wo = [result_wo['accuracy'], result_wo['macro_f1'], result_wo['weighted_f1']]
    vals_w = [result_w['accuracy'], result_w['macro_f1'], result_w['weighted_f1']]
    x = np.arange(len(metrics))
    ax.bar(x - 0.15, vals_wo, 0.3, label='无 Age', color='#4A90D9', edgecolor='black')
    ax.bar(x + 0.15, vals_w, 0.3, label='含 Age', color='#E67E22', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('分数')
    ax.set_title('Age 对照实验 — 总体指标对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'overall_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/overall_comparison.png")


# ==================== 6. 保存结果 ====================

def save_results(result_wo, result_w):
    """保存 CSV 结果文件"""
    pd.DataFrame(result_wo['report']).transpose().to_csv(
        os.path.join(RESULT_DIR, 'classification_report_wo_age.csv'), encoding='utf-8-sig')
    pd.DataFrame(result_w['report']).transpose().to_csv(
        os.path.join(RESULT_DIR, 'classification_report_w_age.csv'), encoding='utf-8-sig')
    pd.DataFrame(result_wo['confusion_matrix'], index=POS_ORDER, columns=POS_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'confusion_matrix_wo_age.csv'), encoding='utf-8-sig')
    pd.DataFrame(result_w['confusion_matrix'], index=POS_ORDER, columns=POS_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'confusion_matrix_w_age.csv'), encoding='utf-8-sig')
    pd.DataFrame({
        '指标': ['Accuracy', 'Macro_F1', 'Weighted_F1'],
        '无Age': [result_wo['accuracy'], result_wo['macro_f1'], result_wo['weighted_f1']],
        '含Age': [result_w['accuracy'], result_w['macro_f1'], result_w['weighted_f1']],
    }).to_csv(os.path.join(RESULT_DIR, 'age_comparison.csv'), index=False, encoding='utf-8-sig')
    print("已保存全部 CSV 到 result/")


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("朴素贝叶斯分类实验 — NBA 球员位置预测")
    print("=" * 60)

    # 1. 加载数据
    global df
    df = load_and_clean_data()

    # 2. 准备特征
    X_wo, X_w, y, feat_wo, feat_w = prepare_data(df)

    # 3. 划分
    X_wo_train, X_wo_test, y_train, y_test = split_data(X_wo, y)
    X_w_train, X_w_test, _, _ = split_data(X_w, y)

    # 4. 训练和评估
    print("\n" + "=" * 60)
    print("4. 模型训练与评估")
    print("=" * 60)
    result_wo = train_and_evaluate(X_wo_train, X_wo_test, y_train, y_test, feat_wo, "NB-A（无 Age，8 维）")
    result_w = train_and_evaluate(X_w_train, X_w_test, y_train, y_test, feat_w, "NB-B（含 Age，9 维）")

    # 5. Age 对照分析
    print("\n" + "=" * 60)
    print("5. Age 对照实验分析")
    print("=" * 60)
    print(f"{'指标':<20} {'无 Age':<15} {'含 Age':<15} {'差异'}")
    print("-" * 55)
    for name, v_wo, v_w in [('Accuracy', result_wo['accuracy'], result_w['accuracy']),
                             ('Macro F1', result_wo['macro_f1'], result_w['macro_f1']),
                             ('Weighted F1', result_wo['weighted_f1'], result_w['weighted_f1'])]:
        diff = v_w - v_wo
        arrow = '▲' if diff > 0 else '▼'
        print(f"{name:<20} {v_wo:<15.4f} {v_w:<15.4f} {arrow} {abs(diff):.4f}")

    # 6. 选最优模型
    best_result = result_w if result_w['accuracy'] >= result_wo['accuracy'] else result_wo
    best_name = "含 Age" if result_w['accuracy'] >= result_wo['accuracy'] else "无 Age"
    print(f"\n最优模型: {best_name}, 准确率={best_result['accuracy']:.4f}")

    # 7. 画图
    print("\n" + "=" * 60)
    print("6. 可视化")
    print("=" * 60)
    draw_confusion_matrix_comparison(result_wo, result_w)
    draw_f1_comparison(result_wo, result_w)
    draw_other_plots(best_result, best_name)
    draw_overall_comparison(result_wo, result_w)

    # 8. 保存结果
    print("\n" + "=" * 60)
    print("7. 保存结果")
    print("=" * 60)
    save_results(result_wo, result_w)

    # 9. 打印结论
    print(f"\n{'=' * 60}")
    print(f"实验完成！")
    print(f"{'=' * 60}")
    print(f"\nAge 影响: {result_w['accuracy'] - result_wo['accuracy']:+.4f}")
    print(f"结论: {'Age 有正向贡献' if result_w['accuracy'] > result_wo['accuracy'] else 'Age 贡献不明显'}")


if __name__ == '__main__':
    df = None
    main()
