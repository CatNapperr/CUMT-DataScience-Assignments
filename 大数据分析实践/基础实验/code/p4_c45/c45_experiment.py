"""
实验四：C4.5 决策树 — NBA 球员位置预测（原始版本）
=====================================================
C4.5 改进（相对于 ID3）：
  1. 增益率（Gain Ratio）替代信息增益，惩罚多值特征
  2. 连续特征自动寻找最优分裂阈值（无需手动离散化）
  3. 悲观后剪枝（Pessimistic Pruning）控制过拟合

预处理：场均统计，16 维特征（含 3PA_G、AST_TOV、FTr、STL_BLK 等工程特征）
使用 KBinsDiscretizer 等频离散化后输入 C4.5
"""
import os
import json
import math
import copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================================================
# 路径配置
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'NBA_Season_Stats.csv')
PIC_DIR = os.path.join(BASE_DIR, 'pic')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ================================================================
# C4.5 决策树核心实现
# ================================================================

class C45DecisionTree:
    """
    C4.5 决策树手动实现
    支持：连续值自动分裂、增益率、悲观后剪枝
    """

    def __init__(self, min_samples=25, prune_confidence=0.25):
        self.tree = None               # 决策树根节点
        self.feature_names = None      # 特征名列表
        self.depth = 0                 # 树的最大深度
        self.min_samples = min_samples          # 最小分裂样本数（预剪枝）
        self.prune_confidence = prune_confidence  # 剪枝置信度参数

    def _entropy(self, labels):
        """计算经验熵 H(D) = -Σ(p_k * log2(p_k))"""
        total = len(labels)
        if total == 0:
            return 0
        counter = Counter(labels)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _split_info(self, left_count, right_count, total):
        """计算分裂信息 SplitInfo = -Σ(|Di|/|D| * log2(|Di|/|D|))"""
        if total == 0:
            return 0
        si = 0.0
        for count in (left_count, right_count):
            if count > 0:
                p = count / total
                si -= p * math.log2(p)
        return si

    def _gain_ratio(self, ent_parent, left_labels, right_labels):
        """
        计算增益率 GainRatio = Gain / SplitInfo
        Gain = Ent(D) - (|D_left|/|D| * Ent(D_left) + |D_right|/|D| * Ent(D_right))
        """
        total = len(left_labels) + len(right_labels)
        if total == 0:
            return 0

        left_ent = self._entropy(left_labels)
        right_ent = self._entropy(right_labels)
        gain = ent_parent - (len(left_labels) / total * left_ent + len(right_labels) / total * right_ent)

        # 分裂信息
        si = self._split_info(len(left_labels), len(right_labels), total)

        # 防止除零：si=0 表示没有有效分裂（所有样本都在同一侧）
        if si < 1e-10:
            return 0
        return gain / si

    def _find_best_continuous_split(self, X_col, labels):
        """
        对单个连续特征搜索最优分裂阈值
        返回: dict 或 None
        """
        n = len(X_col)
        if n < 2:
            return None

        # 将特征值和标签配对并按特征值排序
        paired = sorted(zip(X_col, labels), key=lambda x: x[0])
        sorted_vals = np.array([p[0] for p in paired])
        sorted_labels = [p[1] for p in paired]

        ent_parent = self._entropy(labels)

        best_gr = 0
        best_threshold = None
        best_left_labels = None
        best_right_labels = None

        # 遍历相邻值之间的中点作为候选阈值
        for i in range(1, n):
            # 跳过特征值相同的相邻对
            if sorted_vals[i] == sorted_vals[i - 1]:
                continue

            # 候选阈值 = 相邻两个值的中间点
            threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0

            left_labels = sorted_labels[:i]
            right_labels = sorted_labels[i:]

            # 如果某侧为空，跳过
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            gr = self._gain_ratio(ent_parent, left_labels, right_labels)

            if gr > best_gr:
                best_gr = gr
                best_threshold = threshold
                best_left_labels = left_labels
                best_right_labels = right_labels

        if best_threshold is None or best_gr <= 0:
            return None

        return {
            'threshold': best_threshold,
            'gain_ratio': best_gr,
            'left_labels': best_left_labels,
            'right_labels': best_right_labels,
            'left_count': len(best_left_labels),
            'right_count': len(best_right_labels),
        }

    def _find_best_split(self, X, labels, feature_indices):
        """
        在所有特征中寻找最优分裂（增益率最大）
        返回: dict 或 None
        """
        n = len(labels)
        if n < 2:
            return None

        best_split = None
        best_gr = 0

        for idx in feature_indices:
            X_col = X[:, idx]
            result = self._find_best_continuous_split(X_col, labels)
            if result is not None and result['gain_ratio'] > best_gr:
                best_gr = result['gain_ratio']
                # 记录划分索引
                paired = list(zip(X_col, range(n)))
                left_idx = [i for val, i in paired if val <= result['threshold']]
                right_idx = [i for val, i in paired if val > result['threshold']]
                best_split = {
                    'feature_idx': idx,
                    'feature_name': self.feature_names[idx],
                    'threshold': result['threshold'],
                    'gain_ratio': result['gain_ratio'],
                    'left_indices': left_idx,
                    'right_indices': right_idx,
                }

        if best_split is None or best_split['gain_ratio'] <= 0:
            return None

        return best_split

    def _build_tree(self, X, labels, feature_indices, depth=0):
        """递归构建 C4.5 决策树"""
        self.depth = max(self.depth, depth)

        # 终止条件 1：所有样本属于同一类别
        if len(set(labels)) == 1:
            return {'type': 'leaf', 'label': labels[0], 'count': len(labels)}

        # 终止条件 2：没有可用特征
        if len(feature_indices) == 0:
            majority = Counter(labels).most_common(1)[0][0]
            return {'type': 'leaf', 'label': majority, 'count': len(labels)}

        # 终止条件 3：样本数少于最小分裂阈值（预剪枝）
        if len(labels) < self.min_samples:
            majority = Counter(labels).most_common(1)[0][0]
            return {'type': 'leaf', 'label': majority, 'count': len(labels)}

        # 寻找最优分裂
        best_split = self._find_best_split(X, labels, feature_indices)

        if best_split is None:
            majority = Counter(labels).most_common(1)[0][0]
            return {'type': 'leaf', 'label': majority, 'count': len(labels)}

        # 构建当前节点
        tree_node = {
            'type': 'split',
            'feature_idx': best_split['feature_idx'],
            'feature_name': best_split['feature_name'],
            'threshold': best_split['threshold'],
            'gain_ratio': round(best_split['gain_ratio'], 4),
            'count': len(labels),
            'distribution': dict(Counter(labels)),
            'branches': {},
        }

        # 递归构建左右子树
        remaining = [i for i in feature_indices if i != best_split['feature_idx']]

        # 左分支：<= threshold
        left_X = X[best_split['left_indices']]
        left_labels = [labels[i] for i in best_split['left_indices']]
        tree_node['branches']['<='] = self._build_tree(
            left_X, left_labels, remaining, depth + 1
        )

        # 右分支：> threshold
        right_X = X[best_split['right_indices']]
        right_labels = [labels[i] for i in best_split['right_indices']]
        tree_node['branches']['>'] = self._build_tree(
            right_X, right_labels, remaining, depth + 1
        )

        return tree_node

    # ======== 悲观后剪枝 ========

    def _pessimistic_prune(self, node, X, labels, feature_indices):
        """
        悲观后剪枝（后序遍历，自底向上）
        使用训练集估计误差率 + 连续性校正
        """
        if node['type'] == 'leaf':
            return node

        # 获取当前节点的特征和阈值
        feat_idx = node['feature_idx']
        threshold = node['threshold']
        X_col = X[:, feat_idx]

        # 划分数据
        left_mask = X_col <= threshold
        right_mask = X_col > threshold
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]

        # 递归剪枝子树
        if '<=' in node['branches']:
            node['branches']['<='] = self._pessimistic_prune(
                node['branches']['<='],
                X[left_indices],
                [labels[i] for i in left_indices],
                [i for i in feature_indices if i != feat_idx],
            )
        if '>' in node['branches']:
            node['branches']['>'] = self._pessimistic_prune(
                node['branches']['>'],
                X[right_indices],
                [labels[i] for i in right_indices],
                [i for i in feature_indices if i != feat_idx],
            )

        # ===== 判断是否剪枝 =====
        total = len(labels)
        if total == 0:
            return node

        counter = Counter(labels)
        majority_class = counter.most_common(1)[0][0]
        majority_count = counter.most_common(1)[0][1]
        leaf_errors = total - majority_count
        # 连续性校正：每个叶子加 0.5
        leaf_error_rate = (leaf_errors + 0.5) / total

        # 计算子树的误差估计
        subtree_errors = 0
        subtree_total = 0

        def count_errors(sub_node, sub_X, sub_labels):
            nonlocal subtree_errors, subtree_total
            if sub_node['type'] == 'leaf':
                sub_total = len(sub_labels)
                subtree_total += sub_total
                if sub_total > 0:
                    sub_wrong = sum(1 for l in sub_labels if l != sub_node['label'])
                    subtree_errors += sub_wrong + 0.5  # 每个叶子加 0.5
                return
            sub_feat = sub_node['feature_idx']
            sub_th = sub_node['threshold']
            sub_col = sub_X[:, sub_feat]
            left_m = sub_col <= sub_th
            right_m = sub_col > sub_th
            if '<=' in sub_node['branches']:
                count_errors(sub_node['branches']['<='], sub_X[left_m],
                             [sub_labels[i] for i in np.where(left_m)[0]])
            if '>' in sub_node['branches']:
                count_errors(sub_node['branches']['>'], sub_X[right_m],
                             [sub_labels[i] for i in np.where(right_m)[0]])

        count_errors(node, X, labels)
        subtree_error_rate = subtree_errors / total if total > 0 else float('inf')

        # 如果叶子误差率 <= 子树误差率，则剪枝
        if leaf_error_rate <= subtree_error_rate:
            return {'type': 'leaf', 'label': majority_class, 'count': total}

        return node

    def fit(self, X, labels, feature_names, prune=True):
        """
        训练 C4.5 决策树
        X: 特征矩阵
        labels: 标签列表
        feature_names: 特征名称列表
        prune: 是否进行悲观后剪枝
        """
        self.feature_names = feature_names
        self.depth = 0
        feature_indices = list(range(len(feature_names)))

        # 建树
        self.tree = self._build_tree(X, labels, feature_indices)

        # 后剪枝
        if prune and self.tree is not None:
            self.tree = self._pessimistic_prune(
                self.tree, X, labels, feature_indices
            )

        return self

    # ======== 预测 ========

    def _predict_sample(self, node, sample):
        """预测单个样本"""
        if node['type'] == 'leaf':
            return node['label']

        feat_idx = node['feature_idx']
        threshold = node['threshold']
        val = sample[feat_idx]

        branch_key = '<=' if val <= threshold else '>'
        if branch_key in node['branches']:
            return self._predict_sample(node['branches'][branch_key], sample)
        else:
            # 分支不存在 → 走样本数最多的分支
            best_key = max(node['branches'].keys(),
                           key=lambda k: node['branches'][k].get('count', 0))
            return self._predict_sample(node['branches'][best_key], sample)

    def predict(self, X):
        """预测多个样本"""
        return [self._predict_sample(self.tree, sample) for sample in X]

    # ======== 树统计 ========

    def _tree_size(self, node):
        """计算树的总结点数"""
        if node['type'] == 'leaf':
            return 1
        size = 1
        for branch in node['branches'].values():
            size += self._tree_size(branch)
        return size

    def _leaf_count(self, node):
        """计算叶子节点数"""
        if node['type'] == 'leaf':
            return 1
        count = 0
        for branch in node['branches'].values():
            count += self._leaf_count(branch)
        return count

    def get_tree_stats(self):
        """获取树的统计信息"""
        if self.tree is None:
            return {}

        def calc_depth(node, d=0):
            if node['type'] == 'leaf':
                return d
            return max(calc_depth(b, d+1) for b in node['branches'].values())

        return {
            'depth': calc_depth(self.tree),
            'total_nodes': self._tree_size(self.tree),
            'leaf_nodes': self._leaf_count(self.tree),
        }

    # ======== 打印树结构 ========

    def _print_tree(self, node, indent=0):
        """递归打印树结构"""
        prefix = "  " * indent
        if node['type'] == 'leaf':
            return f"{prefix}└── 预测: {node['label']} (n={node['count']})"
        feat = node['feature_name']
        th = node['threshold']
        lines = [f"{prefix}├── {feat} <= {th:.4f} (GR={node['gain_ratio']})"]
        lines.append(self._print_tree(node['branches']['<='], indent + 1))
        lines.append(f"{prefix}├── {feat} > {th:.4f}")
        lines.append(self._print_tree(node['branches']['>'], indent + 1))
        return "\n".join(lines)

    def print_tree(self):
        """打印整棵决策树"""
        stats = self.get_tree_stats()
        print(f"\nC4.5 决策树结构 (深度={stats['depth']}):")
        print(self._print_tree(self.tree))


# ================================================================
# 数据加载与预处理
# ================================================================

def load_and_filter_data():
    """加载数据，计算场均统计，构造工程特征"""
    df = pd.read_csv(DATA_PATH)

    # 场均统计
    df['PPG'] = df['PTS'] / df['G']
    df['APG'] = df['AST'] / df['G']
    df['RPG'] = df['TRB'] / df['G']
    df['SPG'] = df['STL'] / df['G']
    df['BPG'] = df['BLK'] / df['G']
    df['FPG'] = df['PF'] / df['G']
    df['TOPG'] = df['TOV'] / df['G']

    # 缺失值处理：添加二值标志 + 位置分组中位数填充
    for col in ['3P%', 'FT%']:
        short = col.replace('%', '')
        df[f'Has{short}'] = df[col].notna().astype(int)
        df[col] = df[col].fillna(df.groupby('Pos')[col].transform('median'))
    df['FG%'] = df['FG%'].fillna(df.groupby('Pos')['FG%'].transform('median'))

    # 比率特征（区分位置的核心差异）
    df['3PA_G'] = df['3PA'] / df['G']                         # 三分出手倾向
    df['AST_TOV'] = df['AST'] / df['TOV'].replace(0, np.nan)  # 助攻失误比
    df['AST_TOV'] = df['AST_TOV'].fillna(df['AST_TOV'].median())
    df['FTr'] = df['FTA'] / df['FGA'].replace(0, np.nan)      # 造罚球率
    df['FTr'] = df['FTr'].fillna(0)
    df['STL_BLK'] = (df['STL'] + df['BLK']) / df['G']          # 综合防守活跃度

    # 过滤
    df = df[df['G'] >= 20]
    df = df[df['Tm'] != 'TOT']
    print(f"有效样本: {len(df)} 条")
    return df


# ================================================================
# 全局配置
# ================================================================

FEATURES = [
    'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'FG%', '3P%', 'FT%',
    'FPG', 'TOPG', 'Has3P', 'HasFT', '3PA_G', 'AST_TOV', 'FTr', 'STL_BLK'
]
POS_ORDER_5 = ['C', 'PF', 'SF', 'SG', 'PG']
SUPER_ORDER = ['Big', 'Wing', 'PG']
POS_TO_SUPER = {'C': 'Big', 'PF': 'Big', 'PG': 'PG', 'SF': 'Wing', 'SG': 'Wing'}


# ================================================================
# 实验运行函数
# ================================================================

def run_c45_experiment(df, label_name, class_order, experiment_name):
    """
    运行 C4.5 实验
    df: 数据框
    label_name: 'Pos' 或 'SuperPos'
    class_order: 类别顺序列表
    experiment_name: 实验名称
    """
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"{'='*60}")

    X = df[FEATURES].values.astype(float)

    # 标签编码
    if label_name == 'Pos':
        pos_to_int = {p: i for i, p in enumerate(class_order)}
        y = np.array([pos_to_int[p] for p in df['Pos']])
    else:
        super_to_int = {p: i for i, p in enumerate(class_order)}
        y = np.array([super_to_int[POS_TO_SUPER[p]] for p in df['Pos']])

    # 分层划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 等频离散化：将连续特征转为有序区间，减少噪声干扰
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile',
                                   subsample=200000)
    X_train = discretizer.fit_transform(X_train)
    X_test = discretizer.transform(X_test)

    # ---- 训练 C4.5（无剪枝，对照） ----
    print("\n>> 训练 C4.5（无剪枝）...")
    tree_raw = C45DecisionTree(min_samples=10)
    tree_raw.fit(X_train, y_train.tolist(), FEATURES, prune=False)
    stats_raw = tree_raw.get_tree_stats()
    y_train_pred_raw = tree_raw.predict(X_train)
    y_test_pred_raw = tree_raw.predict(X_test)
    train_acc_raw = accuracy_score(y_train, y_train_pred_raw)
    test_acc_raw = accuracy_score(y_test, y_test_pred_raw)

    # ---- 训练 C4.5（有剪枝） ----
    print(">> 训练 C4.5（有剪枝）...")
    tree_pruned = C45DecisionTree(min_samples=25)
    tree_pruned.fit(X_train, y_train.tolist(), FEATURES, prune=True)
    stats_pruned = tree_pruned.get_tree_stats()
    y_train_pred = tree_pruned.predict(X_train)
    y_test_pred = tree_pruned.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # 详细评估
    report = classification_report(y_test, y_test_pred, target_names=class_order,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    # ---- 打印结果 ----
    print(f"\n  C4.5（无剪枝）:")
    print(f"    训练集准确率: {train_acc_raw:.4f}")
    print(f"    测试集准确率: {test_acc_raw:.4f}")
    print(f"    过拟合差距:   {train_acc_raw - test_acc_raw:.4f}")
    print(f"    树深度: {stats_raw['depth']}, 节点: {stats_raw['total_nodes']}, 叶子: {stats_raw['leaf_nodes']}")

    print(f"\n  C4.5（有剪枝）:")
    print(f"    训练集准确率: {train_acc:.4f}")
    print(f"    测试集准确率: {test_acc:.4f}")
    print(f"    过拟合差距:   {train_acc - test_acc:.4f}")
    print(f"    树深度: {stats_pruned['depth']}, 节点: {stats_pruned['total_nodes']}, 叶子: {stats_pruned['leaf_nodes']}, "
          f"每叶样本: {len(X_train)/stats_pruned['leaf_nodes']:.2f}")

    print(f"\n  测试集分类报告（有剪枝）:")
    print(classification_report(y_test, y_test_pred, target_names=class_order, zero_division=0))

    # 保存结果
    result = {
        'experiment_name': experiment_name,
        'class_order': class_order,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'raw': {
            'train_acc': train_acc_raw,
            'test_acc': test_acc_raw,
            'overfitting_gap': train_acc_raw - test_acc_raw,
            'stats': stats_raw,
        },
        'pruned': {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting_gap': train_acc - test_acc,
            'stats': stats_pruned,
            'report': report,
            'cm': cm,
            'tree': tree_pruned,
        }
    }
    return result


# ================================================================
# 保存结果
# ================================================================

def save_result(result, prefix):
    """保存实验结果到 CSV"""
    # 混淆矩阵
    cm_df = pd.DataFrame(
        result['pruned']['cm'],
        index=result['class_order'],
        columns=result['class_order']
    )
    cm_df.to_csv(os.path.join(RESULT_DIR, f'{prefix}_confusion_matrix.csv'), encoding='utf-8-sig')

    # 摘要
    s = result['pruned']['stats']
    summary = pd.DataFrame([{
        '实验': result['experiment_name'],
        '标签方案': '+'.join(result['class_order']),
        '特征维度': len(FEATURES),
        '训练样本': len(result['y_test']) * 7 // 3,
        '测试样本': len(result['y_test']),
        '训练集准确率(无剪枝)': round(result['raw']['train_acc'], 4),
        '测试集准确率(无剪枝)': round(result['raw']['test_acc'], 4),
        '训练集准确率(有剪枝)': round(result['pruned']['train_acc'], 4),
        '测试集准确率(有剪枝)': round(result['pruned']['test_acc'], 4),
        '过拟合差距(无剪枝)': round(result['raw']['overfitting_gap'], 4),
        '过拟合差距(有剪枝)': round(result['pruned']['overfitting_gap'], 4),
        '树深度': s['depth'],
        '总节点数': s['total_nodes'],
        '叶子节点数': s['leaf_nodes'],
        'Macro_F1': round(result['pruned']['report']['macro avg']['f1-score'], 4),
        'Weighted_F1': round(result['pruned']['report']['weighted avg']['f1-score'], 4),
    }])
    summary.to_csv(os.path.join(RESULT_DIR, f'{prefix}_summary.csv'), index=False, encoding='utf-8-sig')


# ================================================================
# 可视化
# ================================================================

def draw_comparison(results_5, results_3):
    """绘制对比图：跨方案对比 + 混淆矩阵 + 剪枝效果"""
    # 图1：四个实验的测试准确率对比条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    labels_x = ['ID3 原始\n(5类+粗箱)', 'ID3 优化\n(3类+细箱)', 'C4.5 (5类)\n有剪枝', 'C4.5 (3类)\n有剪枝']
    values = [
        0.5162,         # ID3 原始 5类
        0.7596,         # ID3 优化 3类+细箱
        results_5['pruned']['test_acc'],
        results_3['pruned']['test_acc'],
    ]
    colors = ['#E67E22', '#4A90D9', '#2ECC71', '#27AE60']
    bars = ax.bar(labels_x, values, color=colors, edgecolor='black', width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('测试集 Accuracy', fontsize=12)
    ax.set_title('C4.5 决策树 — 与其他方案对比', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'comparison_overall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/comparison_overall.png")

    # 图2：混淆矩阵并列
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (result, title) in enumerate([(results_5, 'C4.5 (5位置)'), (results_3, 'C4.5 (3超类)')]):
        ax = axes[idx]
        cm = result['pruned']['cm']
        cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        classes = result['class_order']
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticklabels(classes, fontsize=11)
        ax.set_xlabel('预测', fontsize=11)
        ax.set_ylabel('真实', fontsize=11)
        ax.set_title(f'{title}\nAcc={result["pruned"]["test_acc"]:.4f}', fontsize=12)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                        ha='center', va='center', fontsize=9,
                        color='white' if cm_pct[i, j] > 50 else 'black')
    plt.suptitle('C4.5 决策树 — 混淆矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/confusion_matrix.png")

    # 图3：无剪枝 vs 有剪枝的效果
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    raw_accs = [
        results_5['raw']['test_acc'],
        results_5['raw']['train_acc'],
        results_3['raw']['test_acc'],
        results_3['raw']['train_acc'],
    ]
    pruned_accs = [
        results_5['pruned']['test_acc'],
        results_5['pruned']['train_acc'],
        results_3['pruned']['test_acc'],
        results_3['pruned']['train_acc'],
    ]
    groups = ['5类-测试', '5类-训练', '3类-测试', '3类-训练']
    bars1 = ax.bar(x - width/2, raw_accs, width, label='无剪枝', color='#E74C3C', edgecolor='black')
    bars2 = ax.bar(x + width/2, pruned_accs, width, label='有剪枝', color='#2ECC71', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('C4.5 剪枝效果对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'pruning_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/pruning_effect.png")


# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("C4.5 决策树实验")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_filter_data()

    # 2. 实验 1：五位置分类
    results_5 = run_c45_experiment(
        df, 'Pos', POS_ORDER_5, 'C4.5-5位置'
    )
    save_result(results_5, 'c45_5pos')

    # 3. 实验 2：三超类分类
    results_3 = run_c45_experiment(
        df, 'SuperPos', SUPER_ORDER, 'C4.5-3超类'
    )
    save_result(results_3, 'c45_3class')

    # 4. 汇总对比
    print("\n" + "=" * 60)
    print("汇总对比（有剪枝）")
    print("=" * 60)

    print(f"\n{'指标':<30} {'C4.5 (5位置)':<18} {'C4.5 (3超类)':<18} {'提升'}")
    print("-" * 85)
    rows = [
        ('测试集 Accuracy',
         results_5['pruned']['test_acc'],
         results_3['pruned']['test_acc']),
        ('训练集 Accuracy',
         results_5['pruned']['train_acc'],
         results_3['pruned']['train_acc']),
        ('过拟合差距',
         results_5['pruned']['overfitting_gap'],
         results_3['pruned']['overfitting_gap']),
        ('Macro F1',
         results_5['pruned']['report']['macro avg']['f1-score'],
         results_3['pruned']['report']['macro avg']['f1-score']),
    ]
    for name, v5, v3 in rows:
        diff = v3 - v5
        arrow = '▲' if diff > 0 else '▼'
        print(f"{name:<30} {v5:<18.4f} {v3:<18.4f} {arrow} {abs(diff):.4f}")

    s5 = results_5['pruned']['stats']
    s3 = results_3['pruned']['stats']
    print(f"{'树深度':<30} {s5['depth']:<18} {s3['depth']:<18}")
    print(f"{'叶子节点数':<30} {s5['leaf_nodes']:<18} {s3['leaf_nodes']:<18}")
    print(f"{'每叶样本数':<30} {len(results_5['y_test'])*7//3 / s5['leaf_nodes']:<18.2f} "
          f"{len(results_3['y_test'])*7//3 / s3['leaf_nodes']:<18.2f}")

    # 5. 剪枝效果
    print(f"\n{'='*60}")
    print("剪枝效果")
    print(f"{'='*60}")
    for name, res in [('5位置', results_5), ('3超类', results_3)]:
        r = res['raw']
        p = res['pruned']
        print(f"\n  {name}:")
        print(f"    无剪枝: 训练={r['train_acc']:.4f}, 测试={r['test_acc']:.4f}, "
              f"差距={r['overfitting_gap']:.4f}, 叶子={r['stats']['leaf_nodes']}")
        print(f"    有剪枝: 训练={p['train_acc']:.4f}, 测试={p['test_acc']:.4f}, "
              f"差距={p['overfitting_gap']:.4f}, 叶子={p['stats']['leaf_nodes']}")

    # 6. 跨方案对比
    print(f"\n{'='*60}")
    print("跨方案对比")
    print(f"{'='*60}")
    print(f"\n{'方案':<35} {'Test Acc':<12} {'叶子数':<10} {'每叶样本':<10}")
    print("-" * 70)
    print(f"{'ID3 原始 (5类+粗箱,去Age)':<35} {'0.5162':<12} {'3453':<10} {'2.81':<10}")
    print(f"{'ID3 实验A (3类+粗箱,去Age)':<35} {'0.7483':<12} {'2382':<10} {'4.07':<10}")
    print(f"{'ID3 实验B (3类+细箱,去Age)':<35} {'0.7596':<12} {'3960':<10} {'2.45':<10}")
    print(f"{'C4.5 (5位置, 有剪枝)':<35} {results_5['pruned']['test_acc']:<12.4f} "
          f"{results_5['pruned']['stats']['leaf_nodes']:<10} "
          f"{len(results_5['y_test'])*7//3 / results_5['pruned']['stats']['leaf_nodes']:<10.2f}")
    print(f"{'C4.5 (3超类, 有剪枝)':<35} {results_3['pruned']['test_acc']:<12.4f} "
          f"{results_3['pruned']['stats']['leaf_nodes']:<10} "
          f"{len(results_3['y_test'])*7//3 / results_3['pruned']['stats']['leaf_nodes']:<10.2f}")
    print(f"{'NB (3超类)':<35} {'0.7584':<12} {'N/A':<10} {'N/A':<10}")

    # 7. 可视化
    print("\n>> 生成可视化...")
    draw_comparison(results_5, results_3)

    print(f"\n{'='*60}")
    print("C4.5 实验完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
