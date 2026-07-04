"""
实验：朴素贝叶斯 — 3 超类分类
功能：将 5 个位置合并为 3 个超类（Big/Wing/PG），
      比较 3 超类直接分类与 5→3 映射的效果
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

# 位置和超类定义
POS_ORDER = ['C', 'PF', 'SF', 'SG', 'PG']
SUPER_ORDER = ['Big', 'Wing', 'PG']
POS_TO_SUPER = {'C': 'Big', 'PF': 'Big', 'PG': 'PG', 'SF': 'Wing', 'SG': 'Wing'}


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

    # 填充缺失值
    for col in ['FG%', '3P%', 'FT%']:
        df[col] = df[col].fillna(0)

    # 过滤
    df = df[df['G'] >= 20]
    df = df[df['Tm'] != 'TOT']
    df = df.reset_index(drop=True)

    print(f"过滤后: {len(df)} 条")
    return df


# ==================== 2. 构造超类标签 ====================

def create_superclass_label(df):
    """将 5 位置映射为 3 超类"""
    print("\n" + "=" * 60)
    print("2. 构造 3 超类标签")
    print("=" * 60)

    df['SuperPos'] = df['Pos'].map(POS_TO_SUPER)
    for cls in SUPER_ORDER:
        cnt = (df['SuperPos'] == cls).sum()
        print(f"  {cls}: {cnt} 条 ({cnt / len(df) * 100:.1f}%)")
    return df


# ==================== 3. 准备特征和标签 ====================

def prepare_features_labels(df):
    """构建 5 类和 3 类的特征矩阵和标签"""
    print("\n" + "=" * 60)
    print("3. 准备特征和标签")
    print("=" * 60)

    features = ['PPG', 'APG', 'RPG', 'SPG', 'BPG', 'FG%', '3P%', 'FT%']
    X = df[features].values
    print(f"特征 (8 维): {features}")

    # 5 类标签
    le5 = LabelEncoder()
    le5.fit(POS_ORDER)
    y5 = le5.transform(df['Pos'])

    # 3 超类标签
    le3 = LabelEncoder()
    le3.fit(SUPER_ORDER)
    y3 = le3.transform(df['SuperPos'])

    print(f"5 类分布: {dict(zip(POS_ORDER, np.bincount(y5)))}")
    print(f"3 类分布: {dict(zip(SUPER_ORDER, np.bincount(y3)))}")

    return X, y5, y3, features, le5, le3


# ==================== 4. 划分数据集 ====================

def split_dataset(X, y5, y3):
    """按 7:3 分层采样划分"""
    print("\n" + "=" * 60)
    print("4. 划分训练集/测试集（7:3 分层采样）")
    print("=" * 60)

    X_train, X_test, y5_train, y5_test, y3_train, y3_test = train_test_split(
        X, y5, y3, test_size=0.3, random_state=42, stratify=y5
    )
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    return X_train, X_test, y5_train, y5_test, y3_train, y3_test


# ==================== 5. 训练与评估 ====================

def train_model(X_train, y_train):
    """训练 GaussianNB 模型"""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_names, model_name):
    """评估模型并输出结果"""
    print(f"\n--- {model_name} ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=label_names, zero_division=0)}")

    return {'y_pred': y_pred, 'y_proba': y_proba, 'accuracy': acc,
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'confusion_matrix': cm}


def compute_5to3_mapping(y5_pred, le5, le3, y3_test):
    """将 5 类预测结果映射为 3 超类，并与真实 3 类标签对比"""
    print("\n" + "=" * 60)
    print("5. 5 类 → 3 类映射对比")
    print("=" * 60)

    # 将 5 类预测结果映射为超类
    y5_pred_super = np.array([POS_TO_SUPER[le5.inverse_transform([p])[0]] for p in y5_pred])
    y5_pred_super_enc = le3.transform(y5_pred_super)

    acc_map = accuracy_score(y3_test, y5_pred_super_enc)
    cm_map = confusion_matrix(y3_test, y5_pred_super_enc)

    return {'y_pred_super': y5_pred_super_enc, 'accuracy': acc_map, 'confusion_matrix': cm_map}


# ==================== 6. 画图 ====================

def draw_confusion_matrices(cm_3class, cm_5to3, acc_3class, acc_5to3):
    """画混淆矩阵对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm_data, title in zip(axes,
                                  [cm_3class, cm_5to3],
                                  [f'3 超类 — 直接分类\nAcc={acc_3class:.3f}',
                                   f'5 类 → 3 类映射\nAcc={acc_5to3:.3f}']):
        cm_pct = cm_data.astype(float) / cm_data.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=SUPER_ORDER, yticklabels=SUPER_ORDER, ax=ax, vmin=0, vmax=100)
        ax.set_xlabel('预测超类')
        ax.set_ylabel('真实超类')
        ax.set_title(title)
    plt.suptitle('3 超类分类 — 混淆矩阵对比（%）', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/confusion_matrix.png")


def draw_comparison_bar(vals_5, vals_3):
    """画 5 类 vs 3 类指标对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    x = np.arange(len(metrics))
    ax.bar(x - 0.15, vals_5, 0.3, label='5 位置分类', color='#E67E22', edgecolor='black')
    ax.bar(x + 0.15, vals_3, 0.3, label='3 超类分类', color='#4A90D9', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('分数')
    ax.set_title('5 位置 vs 3 超类 — 分类性能对比')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'comparison_5vs3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/comparison_5vs3.png")


def draw_class_metrics(y_test, y_pred):
    """画各超类的 P/R/F1"""
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
    plt.savefig(os.path.join(PIC_DIR, 'class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/class_metrics.png")


def draw_feature_distribution(df, features):
    """画按超类分组的特征箱线图"""
    colors = ['#E41A1C', '#4DAF4A', '#984EA3']
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    for i, feat in enumerate(features):
        ax = axes.flatten()[i]
        data = [df[df['SuperPos'] == cls][feat].dropna().values for cls in SUPER_ORDER]
        bp = ax.boxplot(data, tick_labels=SUPER_ORDER, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.set_title(feat, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('各超类在不同特征上的分布对比', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'feature_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/feature_distribution.png")


# ==================== 7. 保存结果 ====================

def save_results(res_5, res_3, res_map, features):
    """保存 CSV 结果"""
    pd.DataFrame(
        classification_report(res_5['y_pred'], res_5['y_pred'],  # placeholder
                              target_names=POS_ORDER, output_dict=True)
    ).transpose().to_csv(os.path.join(RESULT_DIR, 'classification_report_5class.csv'), encoding='utf-8-sig')

    pd.DataFrame({
        'metric': ['Accuracy', 'Macro_F1', 'Weighted_F1'],
        '5class': [res_5['accuracy'], res_5['macro_f1'], res_5['weighted_f1']],
        '3class': [res_3['accuracy'], res_3['macro_f1'], res_3['weighted_f1']],
        '5to3_mapped': [res_map['accuracy'], 0, 0],
    }).to_csv(os.path.join(RESULT_DIR, 'comparison_5vs3.csv'), index=False, encoding='utf-8-sig')

    pd.DataFrame(res_3['confusion_matrix'], index=SUPER_ORDER, columns=SUPER_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'confusion_matrix_3class.csv'), encoding='utf-8-sig')

    pd.DataFrame(res_5['confusion_matrix'], index=POS_ORDER, columns=POS_ORDER).to_csv(
        os.path.join(RESULT_DIR, 'confusion_matrix_5class.csv'), encoding='utf-8-sig')

    print("已保存全部 CSV 到 result/")


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("朴素贝叶斯 — 3 超类分类实验")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_clean_data()

    # 2. 创建超类
    df = create_superclass_label(df)

    # 3. 准备数据
    X, y5, y3, features, le5, le3 = prepare_features_labels(df)
    X_train, X_test, y5_train, y5_test, y3_train, y3_test = split_dataset(X, y5, y3)

    # 4. 训练 5 类模型
    print("\n" + "=" * 60)
    print("5. 模型训练与评估")
    print("=" * 60)
    model_5 = train_model(X_train, y5_train)
    res_5 = evaluate_model(model_5, X_test, y5_test, POS_ORDER, "5 位置分类（8 维）")

    # 5. 训练 3 超类模型
    model_3 = train_model(X_train, y3_train)
    res_3 = evaluate_model(model_3, X_test, y3_test, SUPER_ORDER, "3 超类直接分类（8 维）")

    # 6. 5→3 映射
    res_map = compute_5to3_mapping(res_5['y_pred'], le5, le3, y3_test)
    print(f"5 类 → 3 类映射 Accuracy: {res_map['accuracy']:.4f}")
    print(f"3 超类直接分类 Accuracy:  {res_3['accuracy']:.4f}")
    print(f"提升: {res_3['accuracy'] - res_map['accuracy']:+.4f}")

    # 7. 画图
    print("\n" + "=" * 60)
    print("6. 可视化")
    print("=" * 60)
    draw_confusion_matrices(res_3['confusion_matrix'], res_map['confusion_matrix'],
                            res_3['accuracy'], res_map['accuracy'])
    draw_comparison_bar(
        [res_5['accuracy'], res_5['macro_f1'], res_5['weighted_f1']],
        [res_3['accuracy'], res_3['macro_f1'], res_3['weighted_f1']]
    )
    draw_class_metrics(y3_test, res_3['y_pred'])
    draw_feature_distribution(df, features)

    # 8. 保存
    print("\n" + "=" * 60)
    print("7. 保存结果")
    print("=" * 60)
    save_results(res_5, res_3, res_map, features)

    # 9. 打印总结
    print(f"\n{'=' * 60}")
    print(f"实验完成！")
    print(f"{'=' * 60}")
    print(f"\n{'指标':<25} {'5 类原始':<12} {'3 超类':<12} {'提升'}")
    print("-" * 60)
    for name, v5, v3 in [('Accuracy', res_5['accuracy'], res_3['accuracy']),
                          ('Macro F1', res_5['macro_f1'], res_3['macro_f1']),
                          ('Weighted F1', res_5['weighted_f1'], res_3['weighted_f1'])]:
        print(f"{name:<25} {v5:<12.4f} {v3:<12.4f} {v3 - v5:+8.4f}")


if __name__ == '__main__':
    main()
