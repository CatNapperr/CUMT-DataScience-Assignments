"""
实验：ID3 决策树 — NBA 球员位置预测（改进版）
功能：手动实现 ID3 决策树，使用 per-36 归一化数据预测球员位置
预处理：per-36 归一化、TOT 去重、MP/G 过滤
特征：10 维（PTS_per36, AST_per36, TRB_per36, STL_per36, BLK_per36,
              FG%, 3P%, FT%, PF_per36, TOV_per36）
标签：5 位置（C, PF, SF, SG, PG）+ 3 超类（Big, Wing, PG）
"""

import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================== 路径设置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'NBA_Season_Stats.csv')
PIC_DIR = os.path.join(BASE_DIR, 'pics')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 常数定义
POS_ORDER_5 = ['C', 'PF', 'SF', 'SG', 'PG']
SUPER_ORDER = ['Big', 'Wing', 'PG']
POS_TO_SUPER = {'C': 'Big', 'PF': 'Big', 'PG': 'PG', 'SF': 'Wing', 'SG': 'Wing'}
FEATURE_NAMES = ['PTS_per36', 'AST_per36', 'TRB_per36', 'STL_per36', 'BLK_per36',
                 'FG%', '3P%', 'FT%', 'PF_per36', 'TOV_per36']


# ================================================================
# ID3 决策树核心实现
# ================================================================

class ID3DecisionTree:
    """手动实现 ID3 决策树，仅支持离散型特征"""

    def __init__(self):
        self.tree = None
        self.feature_names = None
        self.depth = 0

    def _calc_entropy(self, labels):
        """计算经验熵 H(D) = -Σ(p_k * log2(p_k))"""
        total = len(labels)
        if total == 0:
            return 0
        counter = Counter(labels)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _calc_cond_entropy(self, feature, labels):
        """计算经验条件熵 H(D|A)"""
        total = len(labels)
        if total == 0:
            return 0
        groups = {}
        for f_val, label in zip(feature, labels):
            groups.setdefault(f_val, []).append(label)
        cond_entropy = 0.0
        for f_val, group_labels in groups.items():
            p = len(group_labels) / total
            cond_entropy += p * self._calc_entropy(group_labels)
        return cond_entropy

    def _calc_info_gain(self, feature, labels):
        """计算信息增益 g(D, A) = H(D) - H(D|A)"""
        return self._calc_entropy(labels) - self._calc_cond_entropy(feature, labels)

    def _choose_best_feature(self, X, labels, feature_indices):
        """选择信息增益最大的特征"""
        best_idx = None
        best_gain = -1
        for idx in feature_indices:
            gain = self._calc_info_gain(X[:, idx], labels)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        return best_idx, best_gain

    def _build_tree(self, X, labels, feature_indices, depth=0):
        """递归构建 ID3 决策树"""
        self.depth = max(self.depth, depth)

        # 终止条件1：所有样本同一类别
        if len(set(labels)) == 1:
            return {'type': 'leaf', 'label': labels[0], 'count': len(labels)}

        # 终止条件2：无可用特征
        if len(feature_indices) == 0:
            majority = Counter(labels).most_common(1)[0][0]
            return {'type': 'leaf', 'label': majority, 'count': len(labels)}

        # 选择最佳分裂特征
        best_idx, best_gain = self._choose_best_feature(X, labels, feature_indices)
        best_feature_name = self.feature_names[best_idx]

        # 终止条件3：信息增益为 0
        if best_gain <= 0:
            majority = Counter(labels).most_common(1)[0][0]
            return {'type': 'leaf', 'label': majority, 'count': len(labels)}

        feature_values = np.unique(X[:, best_idx])
        tree_node = {
            'type': 'split', 'feature_idx': best_idx,
            'feature_name': best_feature_name, 'info_gain': round(best_gain, 4),
            'branches': {}, 'count': len(labels),
            'distribution': dict(Counter(labels))
        }

        remaining_indices = [i for i in feature_indices if i != best_idx]

        for val in feature_values:
            mask = X[:, best_idx] == val
            if np.sum(mask) == 0:
                continue
            X_sub = X[mask]
            labels_sub = [labels[i] for i in range(len(labels)) if mask[i]]
            if len(labels_sub) == 0:
                majority = Counter(labels).most_common(1)[0][0]
                tree_node['branches'][str(val)] = {'type': 'leaf', 'label': majority, 'count': 0}
            else:
                tree_node['branches'][str(val)] = self._build_tree(
                    X_sub, labels_sub, remaining_indices, depth + 1
                )
        return tree_node

    def fit(self, X, labels, feature_names):
        """训练 ID3 决策树"""
        self.feature_names = feature_names
        self.tree = self._build_tree(X, labels, list(range(len(feature_names))))
        return self

    def _predict_sample(self, tree_node, sample):
        """预测单个样本"""
        if tree_node['type'] == 'leaf':
            return tree_node['label']
        feature_idx = tree_node['feature_idx']
        feature_val = str(sample[feature_idx])
        if feature_val in tree_node['branches']:
            return self._predict_sample(tree_node['branches'][feature_val], sample)
        else:
            best_branch = max(tree_node['branches'].values(), key=lambda b: b.get('count', 0))
            return self._predict_sample(best_branch, sample)

    def predict(self, X):
        """预测多个样本"""
        return [self._predict_sample(self.tree, sample) for sample in X]

    def _tree_size(self, node):
        if node['type'] == 'leaf':
            return 1
        size = 1
        for branch in node['branches'].values():
            size += self._tree_size(branch)
        return size

    def _leaf_count(self, node):
        if node['type'] == 'leaf':
            return 1
        count = 0
        for branch in node['branches'].values():
            count += self._leaf_count(branch)
        return count

    def get_tree_stats(self):
        if self.tree is None:
            return {}
        return {
            'depth': self.depth,
            'total_nodes': self._tree_size(self.tree),
            'leaf_nodes': self._leaf_count(self.tree),
        }

    def _print_tree(self, node, indent=0):
        prefix = "  " * indent
        if node['type'] == 'leaf':
            return f"{prefix}└── 预测: {node['label']} (n={node['count']})"
        lines = [f"{prefix}[{node['feature_name']}] IG={node['info_gain']}"]
        for val, branch in node['branches'].items():
            lines.append(f"{prefix}  ├── ={val}:")
            lines.append(self._print_tree(branch, indent + 2))
        return "\n".join(lines)

    def print_tree(self):
        print(f"\n决策树结构 (深度={self.depth}):")
        print(self._print_tree(self.tree))

    def _extract_rules(self, node, current_rule, rules):
        if node['type'] == 'leaf':
            rules.append((current_rule[:], node['label'], node['count']))
            return
        feat_name = node['feature_name']
        for val, branch in node['branches'].items():
            current_rule.append(f"{feat_name}={val}")
            self._extract_rules(branch, current_rule, rules)
            current_rule.pop()

    def get_rules(self):
        rules = []
        self._extract_rules(self.tree, [], rules)
        return rules

    def print_rules(self, top_k=None):
        rules = self.get_rules()
        if top_k:
            rules = rules[:top_k]
        print(f"\n分类规则 (共 {len(self.get_rules())} 条):")
        print("=" * 80)
        for i, (conditions, label, count) in enumerate(rules):
            rule_str = " AND ".join(conditions)
            print(f"  IF {rule_str} THEN Pos={label}  (n={count})")


# ================================================================
# 数据预处理
# ================================================================

def preprocess_data():
    """per-36 预处理，TOT 去重，MP/G 过滤"""
    print("=" * 60)
    print("1. 数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # Per-36 衍生
    per36_map = {
        'PTS_per36': 'PTS', 'AST_per36': 'AST', 'TRB_per36': 'TRB',
        'STL_per36': 'STL', 'BLK_per36': 'BLK', 'PF_per36': 'PF',
        'TOV_per36': 'TOV',
    }
    for new_f, raw_f in per36_map.items():
        df[new_f] = df[raw_f] / df['MP'].replace(0, np.nan) * 36

    # 缺失值填充
    for col in ['FG%', '3P%', 'FT%']:
        df[col] = df[col].fillna(0)

    # TOT 去重
    has_tot = df.groupby(['Year', 'Player'])['Tm'].transform(lambda x: (x == 'TOT').any())
    df = df[~(has_tot & (df['Tm'] != 'TOT'))].copy()

    # 行过滤
    df = df[df['MP'] >= 200]
    df = df[df['G'] >= 10]
    df = df.reset_index(drop=True)

    print(f"预处理后: {len(df)} 条")
    for pos in POS_ORDER_5:
        cnt = (df['Pos'] == pos).sum()
        print(f"  {pos}: {cnt}")

    return df


# ================================================================
# 特征离散化与编码
# ================================================================

def discretize_features(df):
    """对 10 个特征进行离散化分箱"""
    print("\n" + "=" * 60)
    print("2. 特征离散化（10 维分箱）")
    print("=" * 60)

    discretize_rules = {
        'PTS_per36': {
            'bins': [-np.inf, 8, 12, 16, 20, 25, np.inf],
            'labels': ['极低(<8)', '低(8-12)', '中低(12-16)', '中(16-20)', '高(20-25)', '极高(25+)']
        },
        'AST_per36': {
            'bins': [-np.inf, 2, 4, 7, np.inf],
            'labels': ['低(<2)', '中(2-4)', '高(4-7)', '极高(7+)']
        },
        'TRB_per36': {
            'bins': [-np.inf, 4, 7, 11, np.inf],
            'labels': ['低(<4)', '中(4-7)', '高(7-11)', '极高(11+)']
        },
        'STL_per36': {
            'bins': [-np.inf, 0.6, 1.0, 1.5, np.inf],
            'labels': ['极低(<0.6)', '低(0.6-1.0)', '中(1.0-1.5)', '高(1.5+)']
        },
        'BLK_per36': {
            'bins': [-np.inf, 0.3, 0.8, 1.5, np.inf],
            'labels': ['低(<0.3)', '中(0.3-0.8)', '高(0.8-1.5)', '极高(1.5+)']
        },
        'FG%': {
            'bins': [-np.inf, 0.42, 0.47, 0.52, np.inf],
            'labels': ['低(<0.42)', '中(0.42-0.47)', '高(0.47-0.52)', '极高(0.52+)']
        },
        '3P%': {
            'bins': [-np.inf, 0.1, 0.28, 0.36, np.inf],
            'labels': ['无(<0.1)', '低(0.1-0.28)', '中(0.28-0.36)', '高(0.36+)']
        },
        'FT%': {
            'bins': [-np.inf, 0.65, 0.75, 0.83, np.inf],
            'labels': ['差(<0.65)', '中(0.65-0.75)', '好(0.75-0.83)', '优秀(0.83+)']
        },
        'PF_per36': {
            'bins': [-np.inf, 2.5, 3.5, 5, np.inf],
            'labels': ['低(<2.5)', '中(2.5-3.5)', '高(3.5-5)', '极高(5+)']
        },
        'TOV_per36': {
            'bins': [-np.inf, 1.5, 2.0, 3.0, np.inf],
            'labels': ['低(<1.5)', '中(1.5-2.0)', '高(2.0-3.0)', '极高(3+)']
        },
    }

    df_binned = df.copy()
    for feat, rule in discretize_rules.items():
        col = feat + '_bin'
        df_binned[col] = pd.cut(df[feat], bins=rule['bins'], labels=rule['labels'], right=False)
        dist = df_binned[col].value_counts()
        print(f"  {feat:12s} → {len(rule['labels'])} 类: ", end='')
        print(', '.join(f"{l}({dist.get(l,0)})" for l in rule['labels']))

    return df_binned


def encode_features(df_binned):
    """将离散化特征编码为整数矩阵"""
    feature_cols = [f + '_bin' for f in FEATURE_NAMES]
    X_list = []
    for col in feature_cols:
        categories = df_binned[col].cat.categories
        cat_to_int = {cat: i for i, cat in enumerate(categories)}
        X_list.append(df_binned[col].map(cat_to_int).values)
    X = np.column_stack(X_list)

    feature_names_display = [f + '_bin' for f in FEATURE_NAMES]
    return X, feature_names_display


def encode_labels(df_binned, label_mode='5class'):
    """根据标签模式编码 y"""
    if label_mode == '5class':
        class_order = POS_ORDER_5
        class_to_int = {p: i for i, p in enumerate(class_order)}
        y = np.array([class_to_int[p] for p in df_binned['Pos']])
    else:
        class_order = SUPER_ORDER
        class_to_int = {p: i for i, p in enumerate(class_order)}
        y = np.array([class_to_int[POS_TO_SUPER[p]] for p in df_binned['Pos']])
    return y, class_order


# ================================================================
# 单次实验运行
# ================================================================

def run_id3_experiment(X_train, X_test, y_train, y_test, class_order, label_mode, experiment_name):
    """运行单次 ID3 实验，返回结果字典"""
    print(f"\n{'=' * 60}")
    print(f"实验: {experiment_name}")
    print(f"{'=' * 60}")

    display_names = [f.replace('_bin', '') for f in FEATURE_NAMES]
    tree = ID3DecisionTree()
    tree.fit(X_train, y_train.tolist(), display_names)

    stats = tree.get_tree_stats()
    print(f"树深度: {stats['depth']}")
    print(f"总结点数: {stats['total_nodes']}")
    print(f"叶子节点数: {stats['leaf_nodes']}")

    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = train_acc - test_acc

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"过拟合差距:   {gap:.4f}")

    print(f"\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=class_order, zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n混淆矩阵:")
    print(f"{'':>6}", end='')
    for p in class_order:
        print(f"{p:>7}", end='')
    print()
    for i, p in enumerate(class_order):
        print(f"{p:>6}", end='')
        for j in range(len(class_order)):
            print(f"{cm[i,j]:>7}", end='')
        print()

    return {
        'experiment_name': experiment_name,
        'label_mode': label_mode,
        'class_order': class_order,
        'tree': tree,
        'stats': stats,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'cm': cm,
    }


# ================================================================
# 可视化
# ================================================================

def draw_confusion_matrix(cm, class_order, test_acc, prefix, title_suffix):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(9, 8))
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(class_order, fontsize=11)
    ax.set_yticklabels(class_order, fontsize=11)
    ax.set_xlabel('预测位置')
    ax.set_ylabel('真实位置')
    ax.set_title(f'ID3 {title_suffix}混淆矩阵 (Acc={test_acc:.3f})')
    for i in range(len(class_order)):
        for j in range(len(class_order)):
            text = f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=9, color='white' if cm_pct[i,j] > 50 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'confusion_matrix_{prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/confusion_matrix_{prefix}.png")


def draw_feature_importance(tree, prefix):
    """绘制特征重要性（信息增益排序）"""
    feature_gains = {}
    def collect_gains(node):
        if node['type'] == 'split':
            fname = node['feature_name']
            feature_gains.setdefault(fname, []).append(node['info_gain'])
            for branch in node['branches'].values():
                collect_gains(branch)
    collect_gains(tree.tree)

    avg_gains = {k: np.mean(v) for k, v in feature_gains.items()}
    sorted_features = sorted(avg_gains.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [f[0] for f in sorted_features]
    gains = [f[1] for f in sorted_features]
    colors_feat = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), gains, color=colors_feat, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('信息增益 (Information Gain)')
    ax.set_title(f'ID3 特征重要性（{prefix}，信息增益排序）')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    for bar, val in zip(bars, gains):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'feature_importance_{prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/feature_importance_{prefix}.png")
    return sorted_features


def draw_tree_structure(tree, stats, prefix):
    """绘制树结构示意（前3层）"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')
    lines = [f"ID3 决策树 ({prefix}, 深度={stats['depth']}, 节点={stats['total_nodes']}, 叶子={stats['leaf_nodes']})"]
    lines.append("=" * 80)

    def build_text_limited(node, indent=0, max_depth=3):
        pfx = "  " * indent
        if node['type'] == 'leaf' or indent >= max_depth:
            suffix = f" (n={node['count']})" if node['type'] == 'leaf' else f" (子节点 {len(node['branches'])} 个)"
            return [f"{pfx}└── {node.get('feature_name', '')} {node.get('label', '')}{suffix}"]
        txt = [f"{pfx}├── {node['feature_name']} (IG={node['info_gain']})"]
        for val, branch in node['branches'].items():
            txt.append(f"{pfx}│  ├── ={val}:")
            txt.extend(build_text_limited(branch, indent + 2, max_depth))
        return txt

    lines.extend(build_text_limited(tree.tree, max_depth=3))
    lines.append(f"... (完整树共 {stats['total_nodes']} 节点, {stats['leaf_nodes']} 叶子)")

    ax.text(0.01, 0.99, '\n'.join(lines), fontsize=9, fontfamily='Microsoft YaHei',
            va='top', ha='left', transform=ax.transAxes)
    plt.savefig(os.path.join(PIC_DIR, f'tree_structure_{prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/tree_structure_{prefix}.png")


def draw_learning_curve(X_train, y_train, X_test, y_test, display_names, prefix):
    """绘制学习曲线"""
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tree_sizes, leaf_counts, train_accs, test_accs = [], [], [], []

    for ratio in ratios:
        n_samples = int(len(X_train) * ratio)
        X_sub = X_train[:n_samples]
        y_sub = y_train[:n_samples]
        sub_tree = ID3DecisionTree()
        sub_tree.fit(X_sub, y_sub.tolist(), display_names)
        s = sub_tree.get_tree_stats()
        tree_sizes.append(s['total_nodes'])
        leaf_counts.append(s['leaf_nodes'])
        train_accs.append(accuracy_score(y_sub, sub_tree.predict(X_sub)))
        test_accs.append(accuracy_score(y_test, sub_tree.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(ratios, tree_sizes, 'bo-', linewidth=2, markersize=6, label='总节点数')
    axes[0].plot(ratios, leaf_counts, 'rs-', linewidth=2, markersize=6, label='叶子节点数')
    axes[0].set_xlabel('训练集比例'); axes[0].set_ylabel('节点数')
    axes[0].set_title(f'ID3 ({prefix}) 树大小随训练数据变化')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ratios, train_accs, 'g^-', linewidth=2, markersize=6, label='训练集准确率')
    axes[1].plot(ratios, test_accs, 'md-', linewidth=2, markersize=6, label='测试集准确率')
    axes[1].set_xlabel('训练集比例'); axes[1].set_ylabel('准确率')
    axes[1].set_title(f'ID3 ({prefix}) 准确率随训练数据变化')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, f'learning_curve_{prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: pics/learning_curve_{prefix}.png")


def draw_comparison_bar(result_5, result_3):
    """绘制 5 类与 3 类对比条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels_x = ['ID3 (5位置)', 'ID3 (3超类)']
    x = np.arange(len(labels_x))
    width = 0.25
    bars1 = ax.bar(x - width, [result_5['train_acc'], result_3['train_acc']], width,
                   label='训练集', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x, [result_5['test_acc'], result_3['test_acc']], width,
                   label='测试集', color='#2ECC71', edgecolor='black')
    bars3 = ax.bar(x + width, [result_5['gap'], result_3['gap']], width,
                   label='过拟合差距', color='#E74C3C', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('ID3 决策树 — 5位置 vs 3超类对比')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'comparison_5vs3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pics/comparison_5vs3.png")


# ================================================================
# 保存结果
# ================================================================

def save_id3_result(result, sorted_features):
    """保存实验结果"""
    prefix = 'id3_5pos' if result['label_mode'] == '5class' else 'id3_3class'
    class_order = result['class_order']
    tree = result['tree']

    # 混淆矩阵
    pd.DataFrame(result['cm'], index=class_order, columns=class_order).to_csv(
        os.path.join(RESULT_DIR, f'{prefix}_confusion_matrix.csv'), encoding='utf-8-sig')

    # 预测结果
    pd.DataFrame({
        '真实位置': [class_order[i] for i in result['y_test']],
        '预测位置': [class_order[i] for i in result['y_test_pred']],
        '正确': [result['y_test'][i] == result['y_test_pred'][i] for i in range(len(result['y_test']))]
    }).to_csv(os.path.join(RESULT_DIR, f'{prefix}_prediction_results.csv'), index=False, encoding='utf-8-sig')

    # 特征重要性
    if sorted_features is not None:
        pd.DataFrame({
            'feature': [f[0] for f in sorted_features],
            'avg_info_gain': [f[1] for f in sorted_features]
        }).to_csv(os.path.join(RESULT_DIR, f'{prefix}_feature_importance.csv'), index=False, encoding='utf-8-sig')

    # 实验摘要
    pd.DataFrame([{
        'experiment': result['experiment_name'],
        'label_scheme': '+'.join(class_order),
        'features': 10,
        'train_samples': len(result['y_test']) * 7 // 3,
        'test_samples': len(result['y_test']),
        'tree_depth': result['stats']['depth'],
        'total_nodes': result['stats']['total_nodes'],
        'leaf_nodes': result['stats']['leaf_nodes'],
        'train_accuracy': round(result['train_acc'], 4),
        'test_accuracy': round(result['test_acc'], 4),
        'overfitting_gap': round(result['gap'], 4),
        'num_rules': len(tree.get_rules()),
    }]).to_csv(os.path.join(RESULT_DIR, f'{prefix}_summary.csv'), index=False, encoding='utf-8-sig')

    # 规则（仅5类保存）
    if result['label_mode'] == '5class':
        rules = tree.get_rules()
        rules_text = []
        for i, (conditions, label, count) in enumerate(rules):
            rules_text.append(f"Rule{i+1}: IF {' AND '.join(conditions)} THEN Pos={class_order[label]} (samples={count})")
        with open(os.path.join(RESULT_DIR, f'{prefix}_decision_rules.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(rules_text))
        print(f"已保存: result/{prefix}_decision_rules.txt ({len(rules)} 条规则)")


# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("ID3 决策树实验 — NBA 球员位置预测（改进版）")
    print("特征: per-36 + 命中率，10 维")
    print("=" * 60)

    # 1. 数据预处理
    df = preprocess_data()

    # 2. 离散化
    df_binned = discretize_features(df)

    # 3. 编码特征矩阵（与标签无关）
    X, feature_names = encode_features(df_binned)
    display_names = [f.replace('_bin', '') for f in feature_names]
    print(f"\n特征矩阵: {X.shape}, 离散化类别总数: {len(np.unique(X))}")

    # 4. 5 类标签 + 划分
    y_5, pos_order_5 = encode_labels(df_binned, '5class')
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(
        X, y_5, test_size=0.3, random_state=42, stratify=y_5
    )
    print(f"\n5位置: 训练集 {len(X_train_5)} 条, 测试集 {len(X_test_5)} 条")

    # 5. 3 类标签 + 划分
    y_3, super_order = encode_labels(df_binned, '3class')
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
        X, y_3, test_size=0.3, random_state=42, stratify=y_3
    )
    for cls in super_order:
        n_train = (y_train_3 == super_order.index(cls)).sum()
        n_test = (y_test_3 == super_order.index(cls)).sum()
        print(f"  {cls}: 训练={n_train}, 测试={n_test}")

    # ====== 实验1：5 位置分类 ======
    result_5 = run_id3_experiment(
        X_train_5, X_test_5, y_train_5, y_test_5,
        pos_order_5, '5class', 'ID3-5位置'
    )

    # ====== 实验2：3 超类分类 ======
    result_3 = run_id3_experiment(
        X_train_3, X_test_3, y_train_3, y_test_3,
        super_order, '3class', 'ID3-3超类'
    )

    # ====== 可视化 ======
    print("\n" + "=" * 60)
    print("可视化")
    print("=" * 60)

    draw_confusion_matrix(result_5['cm'], pos_order_5, result_5['test_acc'], '5pos', '5位置')
    draw_confusion_matrix(result_3['cm'], super_order, result_3['test_acc'], '3class', '3超类')

    sorted_features_5 = draw_feature_importance(result_5['tree'], '5位置')
    sorted_features_3 = draw_feature_importance(result_3['tree'], '3超类')

    draw_tree_structure(result_5['tree'], result_5['stats'], '5pos')
    draw_tree_structure(result_3['tree'], result_3['stats'], '3class')

    draw_learning_curve(X_train_5, y_train_5, X_test_5, y_test_5, display_names, '5位置')
    draw_learning_curve(X_train_3, y_train_3, X_test_3, y_test_3, display_names, '3超类')

    draw_comparison_bar(result_5, result_3)

    # ====== 保存结果 ======
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    save_id3_result(result_5, sorted_features_5)
    save_id3_result(result_3, sorted_features_3)

    # 汇总摘要
    summary_rows = [
        {
            '实验': 'ID3-5位置', '标签方案': '+'.join(pos_order_5), '特征维度': 10,
            '训练样本': len(X_train_5), '测试样本': len(X_test_5),
            '树深度': result_5['stats']['depth'],
            '总节点数': result_5['stats']['total_nodes'],
            '叶子节点数': result_5['stats']['leaf_nodes'],
            '训练集准确率': round(result_5['train_acc'], 4),
            '测试集准确率': round(result_5['test_acc'], 4),
            '过拟合差距': round(result_5['gap'], 4),
            '分类规则数': len(result_5['tree'].get_rules()),
        },
        {
            '实验': 'ID3-3超类', '标签方案': '+'.join(super_order), '特征维度': 10,
            '训练样本': len(X_train_3), '测试样本': len(X_test_3),
            '树深度': result_3['stats']['depth'],
            '总节点数': result_3['stats']['total_nodes'],
            '叶子节点数': result_3['stats']['leaf_nodes'],
            '训练集准确率': round(result_3['train_acc'], 4),
            '测试集准确率': round(result_3['test_acc'], 4),
            '过拟合差距': round(result_3['gap'], 4),
            '分类规则数': len(result_3['tree'].get_rules()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULT_DIR, 'experiment_summary.csv'),
                                       index=False, encoding='utf-8-sig')

    # ====== 汇总对比 ======
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)

    print(f"\n{'指标':<25} {'ID3 (5位置)':<16} {'ID3 (3超类)':<16} {'变化'}")
    print("-" * 75)
    for name, label in [('test_acc', '测试集准确率'), ('train_acc', '训练集准确率'), ('gap', '过拟合差距')]:
        r5_val = result_5[name]
        r3_val = result_3[name]
        diff = r3_val - r5_val
        arrow = '▲' if diff > 0 else '▼'
        print(f"{label:<25} {r5_val:<16.4f} {r3_val:<16.4f} {arrow} {abs(diff):.4f}")

    print(f"{'树深度':<25} {result_5['stats']['depth']:<16} {result_3['stats']['depth']:<16}")
    print(f"{'叶子节点数':<25} {result_5['stats']['leaf_nodes']:<16} {result_3['stats']['leaf_nodes']:<16}")
    print(f"{'分类规则数':<25} {len(result_5['tree'].get_rules()):<16} {len(result_3['tree'].get_rules()):<16}")

    # ====== 跨方案对比 ======
    print(f"\n{'=' * 60}")
    print("跨方案对比（原始 ID3 vs 新 ID3）")
    print(f"{'=' * 60}")
    print(f"\n{'方案':<35} {'Test Acc':<14} {'叶子数':<10} {'差距':<10}")
    print("-" * 70)
    print(f"{'原始 ID3 (场均, 5位置)':<35} {'0.5162':<14} {'3453':<10} {'0.2557':<10}")
    print(f"{'新 ID3 (per-36, 5位置)':<35} {result_5['test_acc']:<14.4f} "
          f"{result_5['stats']['leaf_nodes']:<10} {result_5['gap']:<10.4f}")
    print(f"{'新 ID3 (per-36, 3超类)':<35} {result_3['test_acc']:<14.4f} "
          f"{result_3['stats']['leaf_nodes']:<10} {result_3['gap']:<10.4f}")
    print(f"{'C4.5 (per-36, 3超类, 剪枝)':<35} {'0.8260':<14} {'32':<10} {'0.0061':<10}")
    print(f"{'NB (per-36, 3超类)':<35} {'0.8265':<14} {'N/A':<10} {'N/A':<10}")

    # ====== ID3 局限性分析 ======
    print(f"\n{'=' * 60}")
    print("ID3 算法局限性分析")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
