"""
实验四：C4.5 决策树 — NBA 球员位置预测（改进版）
====================================================
预处理：per-36 归一化、TOT 去重、MP/G 过滤
特征集：15 维（9 基础 per-36 + 6 风格特征）
  - 基础: PTS_per36, AST_per36, TRB_per36, STL_per36, BLK_per36,
          FG%, FT%, PF_per36, TOV_per36
  - 风格: Has3P, HasFT, 3PA_per36, AST/TOV, FTr, 3PAr
C4.5 改进：增益率、连续特征自动分裂、悲观后剪枝
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================================================
# 路径配置
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'NBA_Season_Stats.csv')
PIC_DIR = os.path.join(BASE_DIR, 'pics')
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
        self.tree = None
        self.feature_names = None
        self.depth = 0
        self.min_samples = min_samples
        self.prune_confidence = prune_confidence

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
        si = self._split_info(len(left_labels), len(right_labels), total)
        # 防止除零
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
            if sorted_vals[i] == sorted_vals[i - 1]:
                continue
            threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
            left_labels = sorted_labels[:i]
            right_labels = sorted_labels[i:]
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
            'threshold': best_threshold, 'gain_ratio': best_gr,
            'left_labels': best_left_labels, 'right_labels': best_right_labels,
            'left_count': len(best_left_labels), 'right_count': len(best_right_labels),
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
                paired = list(zip(X_col, range(n)))
                left_idx = [i for val, i in paired if val <= result['threshold']]
                right_idx = [i for val, i in paired if val > result['threshold']]
                best_split = {
                    'feature_idx': idx, 'feature_name': self.feature_names[idx],
                    'threshold': result['threshold'], 'gain_ratio': result['gain_ratio'],
                    'left_indices': left_idx, 'right_indices': right_idx,
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
            'type': 'split', 'feature_idx': best_split['feature_idx'],
            'feature_name': best_split['feature_name'], 'threshold': best_split['threshold'],
            'gain_ratio': round(best_split['gain_ratio'], 4),
            'count': len(labels), 'distribution': dict(Counter(labels)), 'branches': {},
        }

        remaining = [i for i in feature_indices if i != best_split['feature_idx']]

        # 左分支：<= threshold
        left_X = X[best_split['left_indices']]
        left_labels = [labels[i] for i in best_split['left_indices']]
        tree_node['branches']['<='] = self._build_tree(left_X, left_labels, remaining, depth + 1)

        # 右分支：> threshold
        right_X = X[best_split['right_indices']]
        right_labels = [labels[i] for i in best_split['right_indices']]
        tree_node['branches']['>'] = self._build_tree(right_X, right_labels, remaining, depth + 1)

        return tree_node

    # ======== 悲观后剪枝 ========

    def _pessimistic_prune(self, node, X, labels, feature_indices):
        """
        悲观后剪枝（后序遍历，自底向上）
        使用训练集估计误差率 + 连续性校正
        """
        if node['type'] == 'leaf':
            return node

        feat_idx = node['feature_idx']
        threshold = node['threshold']
        X_col = X[:, feat_idx]

        left_mask = X_col <= threshold
        right_mask = X_col > threshold
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]

        # 递归剪枝子树
        if '<=' in node['branches']:
            node['branches']['<='] = self._pessimistic_prune(
                node['branches']['<='], X[left_indices],
                [labels[i] for i in left_indices],
                [i for i in feature_indices if i != feat_idx],
            )
        if '>' in node['branches']:
            node['branches']['>'] = self._pessimistic_prune(
                node['branches']['>'], X[right_indices],
                [labels[i] for i in right_indices],
                [i for i in feature_indices if i != feat_idx],
            )

        total = len(labels)
        if total == 0:
            return node

        counter = Counter(labels)
        majority_class = counter.most_common(1)[0][0]
        majority_count = counter.most_common(1)[0][1]
        leaf_errors = total - majority_count
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
                    subtree_errors += sub_wrong + 0.5
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
        """训练 C4.5 决策树"""
        self.feature_names = feature_names
        self.depth = 0
        feature_indices = list(range(len(feature_names)))
        self.tree = self._build_tree(X, labels, feature_indices)
        if prune and self.tree is not None:
            self.tree = self._pessimistic_prune(self.tree, X, labels, feature_indices)
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

    def print_tree(self, node=None, indent=0):
        """递归打印决策树结构"""
        if node is None:
            node = self.tree
            print(f"\nC4.5 决策树结构 (深度={self.get_tree_stats()['depth']}):")
        prefix = "  " * indent
        if node['type'] == 'leaf':
            print(f"{prefix}└── 预测: {node['label']} (n={node['count']})")
            return
        feat = node['feature_name']
        th = node['threshold']
        print(f"{prefix}├── {feat} <= {th:.4f} (GR={node['gain_ratio']})")
        self.print_tree(node['branches']['<='], indent + 1)
        print(f"{prefix}├── {feat} > {th:.4f}")
        self.print_tree(node['branches']['>'], indent + 1)

    # ======== 规则提取 ========

    def _extract_rules(self, node, current_rule, rules):
        """递归提取决策规则"""
        if node['type'] == 'leaf':
            rules.append((current_rule[:], node['label'], node['count']))
            return
        feat_name = node['feature_name']
        th = node['threshold']
        for branch_key, branch in node['branches'].items():
            current_rule.append(f"{feat_name}{branch_key}{th:.2f}")
            self._extract_rules(branch, current_rule, rules)
            current_rule.pop()

    def get_rules(self):
        """获取所有决策规则"""
        rules = []
        self._extract_rules(self.tree, [], rules)
        return rules


# ================================================================
# 数据预处理
# ================================================================

def preprocess_data():
    """per-36 预处理（与 Bayes/ID3/KMeans 一致）"""
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # 1) Per-36 衍生
    for new_f, raw_f in {
        'PTS_per36': 'PTS', 'AST_per36': 'AST', 'TRB_per36': 'TRB',
        'STL_per36': 'STL', 'BLK_per36': 'BLK', 'PF_per36': 'PF',
        'TOV_per36': 'TOV', '3PA_per36': '3PA',
    }.items():
        df[new_f] = df[raw_f] / df['MP'].replace(0, np.nan) * 36

    # 2) 比率特征
    df['AST_TOV'] = df['AST'] / df['TOV'].replace(0, np.nan)
    df['FTr'] = df['FTA'] / df['FGA'].replace(0, np.nan)
    df['3PAr'] = df['3PA'] / df['FGA'].replace(0, np.nan)

    # 3) 命中率缺失值填充
    for col in ['FG%', 'FT%']:
        df[col] = df[col].fillna(0)

    # 4) 二值标志
    df['Has3P'] = (df['3PA'] > 0).astype(int)
    df['HasFT'] = (df['FTA'] > 0).astype(int)

    # 5) 比率缺失值填充（除零保护）
    df['AST_TOV'] = df['AST_TOV'].fillna(0)
    df['FTr'] = df['FTr'].fillna(0)
    df['3PAr'] = df['3PAr'].fillna(0)

    # 6) TOT 去重：赛季中转会的球员保留 TOT 行，删除单球队行
    has_tot = df.groupby(['Year', 'Player'])['Tm'].transform(lambda x: (x == 'TOT').any())
    df = df[~(has_tot & (df['Tm'] != 'TOT'))].copy()

    # 7) 行过滤
    df = df[df['MP'] >= 200]
    df = df[df['G'] >= 10]
    df = df.reset_index(drop=True)

    print(f"预处理后: {len(df)} 条")
    for pos in ['C', 'PF', 'SF', 'SG', 'PG']:
        print(f"  {pos}: {(df['Pos'] == pos).sum()}")
    return df


# ================================================================
# 特征与标签配置
# ================================================================

# 基础统计特征（9项）
BASE_FEATURES = [
    'PTS_per36', 'AST_per36', 'TRB_per36', 'STL_per36', 'BLK_per36',
    'FG%', 'FT%', 'PF_per36', 'TOV_per36',
]
# 风格特征（6项）
STYLE_FEATURES = [
    'Has3P', 'HasFT', '3PA_per36', 'AST_TOV', 'FTr', '3PAr',
]
# 总特征（15维）
ALL_FEATURES = BASE_FEATURES + STYLE_FEATURES

POS_ORDER_5 = ['C', 'PF', 'SF', 'SG', 'PG']
SUPER_ORDER = ['Big', 'Wing', 'PG']
POS_TO_SUPER = {'C': 'Big', 'PF': 'Big', 'PG': 'PG', 'SF': 'Wing', 'SG': 'Wing'}


# ================================================================
# 实验运行函数
# ================================================================

def run_c45_experiment(df, label_name, class_order, experiment_name):
    """
    运行 C4.5 实验（5位置 / 3超类）
    df: 数据框
    label_name: 'Pos' 或 'SuperPos'
    class_order: 类别顺序列表
    experiment_name: 实验名称
    """
    print(f"\n{'=' * 60}")
    print(f"实验: {experiment_name}")
    print(f"{'=' * 60}")

    X = df[ALL_FEATURES].values.astype(float)
    print(f"特征矩阵: {X.shape}  ({len(ALL_FEATURES)} 维)")

    # 标签编码
    if label_name == 'Pos':
        class_to_int = {p: i for i, p in enumerate(class_order)}
        y = np.array([class_to_int[p] for p in df['Pos']])
    else:
        class_to_int = {p: i for i, p in enumerate(class_order)}
        y = np.array([class_to_int[POS_TO_SUPER[p]] for p in df['Pos']])

    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
    for cls_name in class_order:
        n_train = (y_train == class_to_int[cls_name]).sum()
        n_test = (y_test == class_to_int[cls_name]).sum()
        print(f"  {cls_name}: 训练={n_train}, 测试={n_test}")

    # ---- 训练 C4.5（无剪枝，对照）----
    print("\n>> 训练 C4.5（无剪枝）...")
    tree_raw = C45DecisionTree(min_samples=10)
    tree_raw.fit(X_train, y_train.tolist(), ALL_FEATURES, prune=False)
    stats_raw = tree_raw.get_tree_stats()
    y_train_pred_raw = tree_raw.predict(X_train)
    y_test_pred_raw = tree_raw.predict(X_test)
    train_acc_raw = accuracy_score(y_train, y_train_pred_raw)
    test_acc_raw = accuracy_score(y_test, y_test_pred_raw)
    gap_raw = train_acc_raw - test_acc_raw

    # ---- 训练 C4.5（有剪枝）----
    print(">> 训练 C4.5（有剪枝）...")
    tree_pruned = C45DecisionTree(min_samples=25)
    tree_pruned.fit(X_train, y_train.tolist(), ALL_FEATURES, prune=True)
    stats_pruned = tree_pruned.get_tree_stats()
    y_train_pred = tree_pruned.predict(X_train)
    y_test_pred = tree_pruned.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = train_acc - test_acc

    # 详细评估
    report = classification_report(y_test, y_test_pred, target_names=class_order,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    # ---- 打印结果 ----
    print(f"\n  C4.5（无剪枝）:")
    print(f"    训练集准确率: {train_acc_raw:.4f}")
    print(f"    测试集准确率: {test_acc_raw:.4f}")
    print(f"    过拟合差距:   {gap_raw:.4f}")
    print(f"    树深度: {stats_raw['depth']}, 节点: {stats_raw['total_nodes']}, 叶子: {stats_raw['leaf_nodes']}")

    print(f"\n  C4.5（有剪枝）:")
    print(f"    训练集准确率: {train_acc:.4f}")
    print(f"    测试集准确率: {test_acc:.4f}")
    print(f"    过拟合差距:   {gap:.4f}")
    print(f"    树深度: {stats_pruned['depth']}, 节点: {stats_pruned['total_nodes']}, 叶子: {stats_pruned['leaf_nodes']}, "
          f"每叶样本: {len(X_train) / stats_pruned['leaf_nodes']:.2f}")

    print(f"\n  测试集分类报告（有剪枝）:")
    print(classification_report(y_test, y_test_pred, target_names=class_order, zero_division=0))

    # 返回结果
    result = {
        'experiment_name': experiment_name,
        'class_order': class_order,
        'class_to_int': class_to_int,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_test_pred,
        'raw': {
            'train_acc': train_acc_raw, 'test_acc': test_acc_raw,
            'overfitting_gap': gap_raw, 'stats': stats_raw,
        },
        'pruned': {
            'train_acc': train_acc, 'test_acc': test_acc,
            'overfitting_gap': gap, 'stats': stats_pruned,
            'report': report, 'cm': cm, 'tree': tree_pruned,
        },
        'X_train': X_train, 'y_train': y_train,
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
        '特征维度': len(ALL_FEATURES),
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
        '每叶样本数': round(len(result['y_test']) * 7 // 3 / s['leaf_nodes'], 2),
        'Macro_F1': round(result['pruned']['report']['macro avg']['f1-score'], 4),
        'Weighted_F1': round(result['pruned']['report']['weighted avg']['f1-score'], 4),
    }])
    summary.to_csv(os.path.join(RESULT_DIR, f'{prefix}_summary.csv'), index=False, encoding='utf-8-sig')


# ================================================================
# 可视化函数
# ================================================================

def draw_confusion_matrix(result, prefix):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(9, 8))
    cm = result['pruned']['cm']
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    classes = result['class_order']
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('预测位置')
    ax.set_ylabel('真实位置')
    ax.set_title(f"C4.5 {result['experiment_name']} 混淆矩阵\nAcc={result['pruned']['test_acc']:.4f}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=9, color='white' if cm_pct[i,j] > 50 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'{prefix}_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/{prefix}_confusion_matrix.png")


def draw_pruning_comparison(results_5, results_3):
    """绘制剪枝效果对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    raw_accs = [
        results_5['raw']['test_acc'], results_5['raw']['train_acc'],
        results_3['raw']['test_acc'], results_3['raw']['train_acc'],
    ]
    pruned_accs = [
        results_5['pruned']['test_acc'], results_5['pruned']['train_acc'],
        results_3['pruned']['test_acc'], results_3['pruned']['train_acc'],
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
    print("已保存: pics/pruning_effect.png")


def draw_learning_curve(X_train, y_train, X_test, y_test, prefix):
    """绘制学习曲线：树大小和准确率随训练数据比例的变化"""
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tree_sizes, leaf_counts, train_accs, test_accs = [], [], [], []
    for ratio in ratios:
        n_samples = int(len(X_train) * ratio)
        X_sub = X_train[:n_samples]
        y_sub = y_train[:n_samples]
        sub_tree = C45DecisionTree(min_samples=25)
        sub_tree.fit(X_sub, y_sub.tolist(), ALL_FEATURES, prune=True)
        s = sub_tree.get_tree_stats()
        tree_sizes.append(s['total_nodes'])
        leaf_counts.append(s['leaf_nodes'])
        train_accs.append(accuracy_score(y_sub, sub_tree.predict(X_sub)))
        test_accs.append(accuracy_score(y_test, sub_tree.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(ratios, tree_sizes, 'bo-', linewidth=2, markersize=6, label='总节点数')
    axes[0].plot(ratios, leaf_counts, 'rs-', linewidth=2, markersize=6, label='叶子节点数')
    axes[0].set_xlabel('训练集比例'); axes[0].set_ylabel('节点数')
    axes[0].set_title('C4.5 树大小随训练数据变化'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ratios, train_accs, 'g^-', linewidth=2, markersize=6, label='训练集准确率')
    axes[1].plot(ratios, test_accs, 'md-', linewidth=2, markersize=6, label='测试集准确率')
    axes[1].set_xlabel('训练集比例'); axes[1].set_ylabel('准确率')
    axes[1].set_title('C4.5 准确率随训练数据变化'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'{prefix}_learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/{prefix}_learning_curve.png")


def draw_comparison_overall(results_5, results_3):
    """绘制整体对比图：测试集准确率对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    labels_x = ['C4.5 (5类)\n无剪枝', 'C4.5 (5类)\n有剪枝',
                'C4.5 (3类)\n无剪枝', 'C4.5 (3类)\n有剪枝']
    values = [
        results_5['raw']['test_acc'],
        results_5['pruned']['test_acc'],
        results_3['raw']['test_acc'],
        results_3['pruned']['test_acc'],
    ]
    colors = ['#E74C3C', '#2ECC71', '#E67E22', '#27AE60']
    bars = ax.bar(labels_x, values, color=colors, edgecolor='black', width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('测试集 Accuracy', fontsize=12)
    ax.set_title('C4.5 决策树实验 — 剪枝效果对比', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'comparison_overall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pics/comparison_overall.png")


# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("C4.5 决策树实验 — NBA 球员位置预测（改进版）")
    print("特征: 15 维（9 基础 per-36 + 6 风格特征）")
    print("=" * 60)

    # 1. 数据预处理
    df = preprocess_data()

    # 2. 描述性统计
    print("\n" + "=" * 60)
    print("2. 特征统计描述")
    print("=" * 60)
    desc = df[ALL_FEATURES].describe().round(3)
    print(desc)
    desc.to_csv(os.path.join(RESULT_DIR, 'feature_statistics.csv'), encoding='utf-8-sig')

    # 3. 实验 1：五位置分类
    results_5 = run_c45_experiment(df, 'Pos', POS_ORDER_5, 'C4.5-5位置')
    save_result(results_5, 'c45_5pos')

    # 4. 实验 2：三超类分类
    results_3 = run_c45_experiment(df, 'SuperPos', SUPER_ORDER, 'C4.5-3超类')
    save_result(results_3, 'c45_3class')

    # 5. 可视化
    print("\n" + "=" * 60)
    print("5. 生成可视化")
    print("=" * 60)

    draw_confusion_matrix(results_5, 'c45_5pos')
    draw_confusion_matrix(results_3, 'c45_3class')
    draw_pruning_comparison(results_5, results_3)
    draw_learning_curve(
        results_5['X_train'], results_5['y_train'],
        results_5['X_test'], results_5['y_test'],
        'c45_5pos'
    )
    draw_comparison_overall(results_5, results_3)

    # 6. 汇总对比
    print("\n" + "=" * 60)
    print("6. 汇总对比（有剪枝）")
    print("=" * 60)

    print(f"\n{'指标':<30} {'C4.5 (5位置)':<18} {'C4.5 (3超类)':<18} {'提升'}")
    print("-" * 85)
    rows = [
        ('测试集 Accuracy', results_5['pruned']['test_acc'], results_3['pruned']['test_acc']),
        ('训练集 Accuracy', results_5['pruned']['train_acc'], results_3['pruned']['train_acc']),
        ('过拟合差距', results_5['pruned']['overfitting_gap'], results_3['pruned']['overfitting_gap']),
        ('Macro F1', results_5['pruned']['report']['macro avg']['f1-score'],
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

    # 7. 剪枝效果总结
    print(f"\n{'=' * 60}")
    print("7. 剪枝效果")
    print(f"{'=' * 60}")
    for name, res in [('5位置', results_5), ('3超类', results_3)]:
        r = res['raw']
        p = res['pruned']
        print(f"\n  {name}:")
        print(f"    无剪枝: 训练={r['train_acc']:.4f}, 测试={r['test_acc']:.4f}, "
              f"差距={r['overfitting_gap']:.4f}, 叶子={r['stats']['leaf_nodes']}")
        print(f"    有剪枝: 训练={p['train_acc']:.4f}, 测试={p['test_acc']:.4f}, "
              f"差距={p['overfitting_gap']:.4f}, 叶子={p['stats']['leaf_nodes']}")

    # 8. 跨方案对比
    print(f"\n{'=' * 60}")
    print("8. 跨方案对比（参考 ID3/Bayes）")
    print(f"{'=' * 60}")
    print(f"\n{'方案':<35} {'Test Acc':<12} {'叶子数':<10} {'每叶样本':<10}")
    print("-" * 70)
    print(f"{'ID3 (10维 per-36, 5类)':<35} {'(待运行)':<12} {'...':<10} {'...':<10}")
    print(f"{'C4.5 (15维 per-36, 5位置, 无剪枝)':<35} {results_5['raw']['test_acc']:<12.4f} "
          f"{results_5['raw']['stats']['leaf_nodes']:<10} "
          f"{len(results_5['y_test']) * 7 // 3 / results_5['raw']['stats']['leaf_nodes']:<10.2f}")
    print(f"{'C4.5 (15维 per-36, 5位置, 有剪枝)':<35} {results_5['pruned']['test_acc']:<12.4f} "
          f"{results_5['pruned']['stats']['leaf_nodes']:<10} "
          f"{len(results_5['y_test']) * 7 // 3 / results_5['pruned']['stats']['leaf_nodes']:<10.2f}")
    print(f"{'C4.5 (15维 per-36, 3超类, 无剪枝)':<35} {results_3['raw']['test_acc']:<12.4f} "
          f"{results_3['raw']['stats']['leaf_nodes']:<10} "
          f"{len(results_3['y_test']) * 7 // 3 / results_3['raw']['stats']['leaf_nodes']:<10.2f}")
    print(f"{'C4.5 (15维 per-36, 3超类, 有剪枝)':<35} {results_3['pruned']['test_acc']:<12.4f} "
          f"{results_3['pruned']['stats']['leaf_nodes']:<10} "
          f"{len(results_3['y_test']) * 7 // 3 / results_3['pruned']['stats']['leaf_nodes']:<10.2f}")
    print(f"{'NB (12维 per-36, 3超类)':<35} {'0.8265':<12} {'N/A':<10} {'N/A':<10}")

    # 9. 打印树结构（仅 3 超类版本，更简洁）
    print(f"\n{'=' * 60}")
    print("9. 决策树结构（3超类-有剪枝）")
    print(f"{'=' * 60}")
    results_3['pruned']['tree'].print_tree()

    # 10. 提取规则保存
    rules = results_3['pruned']['tree'].get_rules()
    rules_text = []
    for i, (conditions, label, count) in enumerate(rules):
        class_name = SUPER_ORDER[label]
        rules_text.append(f"Rule{i+1}: IF {' AND '.join(conditions)} THEN Pos={class_name} (samples={count})")
    with open(os.path.join(RESULT_DIR, 'decision_rules.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rules_text))
    print(f"已保存: result/decision_rules.txt ({len(rules)} 条规则)")

    print(f"\n{'=' * 60}")
    print("C4.5 实验完成！")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
