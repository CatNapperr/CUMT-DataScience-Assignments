"""
实验：ID3 决策树 — NBA 球员位置预测（原始版）
功能：手动实现 ID3 决策树，根据离散化的球员场均统计特征预测场上位置
预处理：场均统计、缺失值填充、G>=20 过滤、删除 TOT 行
特征：10 维场均统计分箱后作为离散特征输入
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
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
PIC_DIR = os.path.join(BASE_DIR, 'pic')
RESULT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

POS_ORDER = ['C', 'PF', 'SF', 'SG', 'PG']

# 分箱规则定义
DISCRETIZE_RULES = {
    'PPG': {
        'bins': [-np.inf, 5, 10, 15, 20, 25, np.inf],
        'labels': ['极低(<5)', '低(5-10)', '中低(10-15)', '中(15-20)', '高(20-25)', '极高(25+)']
    },
    'APG': {
        'bins': [-np.inf, 2, 5, 10, np.inf],
        'labels': ['低(<2)', '中(2-5)', '高(5-10)', '极高(10+)']
    },
    'RPG': {
        'bins': [-np.inf, 3, 6, 10, np.inf],
        'labels': ['低(<3)', '中(3-6)', '高(6-10)', '极高(10+)']
    },
    'SPG': {
        'bins': [-np.inf, 0.5, 1, 2, np.inf],
        'labels': ['极低(<0.5)', '低(0.5-1)', '中(1-2)', '高(2+)']
    },
    'BPG': {
        'bins': [-np.inf, 0.5, 1.5, np.inf],
        'labels': ['低(<0.5)', '中(0.5-1.5)', '高(1.5+)']
    },
    'FG%': {
        'bins': [-np.inf, 0.45, 0.50, np.inf],
        'labels': ['低(<0.45)', '中(0.45-0.50)', '高(0.50+)']
    },
    '3P%': {
        'bins': [-np.inf, 0.1, 0.3, 0.36, np.inf],
        'labels': ['无(<0.1)', '差(0.1-0.3)', '好(0.3-0.36)', '优秀(0.36+)']
    },
    'FT%': {
        'bins': [-np.inf, 0.7, 0.8, np.inf],
        'labels': ['差(<0.7)', '中(0.7-0.8)', '好(0.8+)']
    },
    'FPG': {
        'bins': [-np.inf, 1, 2, 3, np.inf],
        'labels': ['低(<1)', '中(1-2)', '高(2-3)', '极高(3+)']
    },
    'TOPG': {
        'bins': [-np.inf, 1, 2, 3, np.inf],
        'labels': ['低(<1)', '中(1-2)', '高(2-3)', '极高(3+)']
    },
}

FEATURE_LIST = ['PPG', 'APG', 'RPG', 'SPG', 'BPG', 'FG%', '3P%', 'FT%', 'FPG', 'TOPG']


# ================================================================
# ID3 决策树核心实现
# ================================================================

class ID3DecisionTree:
    """
    手动实现 ID3 决策树算法
    仅支持离散型特征，使用信息增益选择分裂属性
    """

    def __init__(self):
        self.tree = None
        self.feature_names = None
        self.label_name = None
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
        """计算经验条件熵 H(D|A) = Σ(|D_i|/|D| * H(D_i))"""
        total = len(labels)
        if total == 0:
            return 0
        groups = {}
        for f_val, label in zip(feature, labels):
            if f_val not in groups:
                groups[f_val] = []
            groups[f_val].append(label)
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
            feature = X[:, idx]
            gain = self._calc_info_gain(feature, labels)
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

        # 终止条件3：信息增益为0
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

    def fit(self, X, labels, feature_names, label_name='Pos'):
        """训练 ID3 决策树"""
        self.feature_names = feature_names
        self.label_name = label_name
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
            # 分支不存在时走多数分支
            best_branch = max(tree_node['branches'].values(), key=lambda b: b.get('count', 0))
            return self._predict_sample(best_branch, sample)

    def predict(self, X):
        """预测多个样本"""
        return [self._predict_sample(self.tree, sample) for sample in X]

    def _tree_size(self, node):
        if node['type'] == 'leaf':
            return 1
        return 1 + sum(self._tree_size(b) for b in node['branches'].values())

    def _leaf_count(self, node):
        if node['type'] == 'leaf':
            return 1
        return sum(self._leaf_count(b) for b in node['branches'].values())

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
            return f"{prefix}└── 预测: {node['label']} (样本数: {node['count']})"
        lines = [f"{prefix}[{node['feature_name']}] IG={node['info_gain']}"]
        for val, branch in node['branches'].items():
            lines.append(f"{prefix}  ├── {node['feature_name']}={val}:")
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
        print(f"\n分类规则 (共 {len(self.get_rules())} 条，显示前 {min(top_k or len(rules), len(rules))} 条):")
        print("=" * 80)
        for i, (conditions, label, count) in enumerate(rules):
            rule_str = " AND ".join(conditions)
            print(f"  IF {rule_str} THEN Pos={label}  (样本数: {count})")


# ================================================================
# 数据预处理
# ================================================================

def preprocess_data():
    """加载数据，计算场均统计，填充缺失值，过滤样本"""
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

    # 缺失值填充
    df['FG%'] = df['FG%'].fillna(0)
    df['3P%'] = df['3P%'].fillna(0)
    df['FT%'] = df['FT%'].fillna(0)

    # 严格过滤
    n_before = len(df)
    df = df[df['G'] >= 20]
    df = df[df['Tm'] != 'TOT']
    print(f"过滤后: {n_before} → {len(df)} 条")
    for pos in POS_ORDER:
        cnt = (df['Pos'] == pos).sum()
        print(f"  {pos}: {cnt} ({cnt/len(df)*100:.1f}%)")

    return df


# ================================================================
# 特征离散化与编码
# ================================================================

def discretize_features(df):
    """对 10 个特征进行离散化分箱"""
    print("\n" + "=" * 60)
    print("2. 特征离散化（10 维分箱）")
    print("=" * 60)

    df_binned = df.copy()
    for feat in FEATURE_LIST:
        rule = DISCRETIZE_RULES[feat]
        df_binned[feat + '_bin'] = pd.cut(
            df[feat], bins=rule['bins'], labels=rule['labels'], right=False
        )
        dist = df_binned[feat + '_bin'].value_counts()
        n_cat = len(rule['labels'])
        print(f"  {feat:6s} → {n_cat} 类: {', '.join(f'{l}({dist.get(l,0)})' for l in rule['labels'])}")

    return df_binned


def encode_features(df_binned):
    """将离散化特征和标签编码为整数"""
    feature_cols = [f + '_bin' for f in FEATURE_LIST]
    feature_names = feature_cols

    X_list = []
    for col in feature_cols:
        categories = df_binned[col].cat.categories
        cat_to_int = {cat: i for i, cat in enumerate(categories)}
        X_list.append(df_binned[col].map(cat_to_int).values)

    X = np.column_stack(X_list)

    # 标签编码
    pos_to_int = {pos: i for i, pos in enumerate(POS_ORDER)}
    y = np.array([pos_to_int[p] for p in df_binned['Pos']])

    return X, y, feature_names, POS_ORDER


# ================================================================
# 训练与评估
# ================================================================

def split_data(X, y):
    """7:3 分层采样划分数据集"""
    print("\n" + "=" * 60)
    print("3. 数据集划分（7:3 分层采样）")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    return X_train, X_test, y_train, y_test


def train_id3(X_train, y_train, feature_names):
    """训练 ID3 决策树"""
    print("\n" + "=" * 60)
    print("4. 训练 ID3 决策树")
    print("=" * 60)

    display_names = [col.replace('_bin', '') for col in feature_names]
    tree = ID3DecisionTree()
    tree.fit(X_train, y_train.tolist(), display_names)

    stats = tree.get_tree_stats()
    print(f"树深度: {stats['depth']}")
    print(f"总结点数: {stats['total_nodes']}")
    print(f"叶子节点数: {stats['leaf_nodes']}")

    return tree, stats


def evaluate_id3(tree, X_train, y_train, X_test, y_test, pos_order):
    """预测并评估模型"""
    print("\n" + "=" * 60)
    print("5. 预测与评估")
    print("=" * 60)

    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"过拟合程度: {train_acc - test_acc:.4f}")

    print(f"\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=pos_order, zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n混淆矩阵:")
    print(f"{'':>6}", end='')
    for p in pos_order:
        print(f"{p:>6}", end='')
    print()
    for i, p in enumerate(pos_order):
        print(f"{p:>6}", end='')
        for j in range(len(pos_order)):
            print(f"{cm[i,j]:>6}", end='')
        print()

    return train_acc, test_acc, y_train_pred, y_test_pred, cm


# ================================================================
# 可视化
# ================================================================

def draw_confusion_matrix(cm, pos_order, test_acc):
    """绘制混淆矩阵热力图"""
    print("绘制: 混淆矩阵...")
    fig, ax = plt.subplots(figsize=(9, 8))
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    ax.set_xticks(range(len(pos_order)))
    ax.set_yticks(range(len(pos_order)))
    ax.set_xticklabels(pos_order, fontsize=11)
    ax.set_yticklabels(pos_order, fontsize=11)
    ax.set_xlabel('预测位置', fontsize=12)
    ax.set_ylabel('真实位置', fontsize=12)
    ax.set_title(f'ID3 决策树混淆矩阵 (Acc={test_acc:.3f})', fontsize=14)

    for i in range(len(pos_order)):
        for j in range(len(pos_order)):
            text = f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=9, color='white' if cm_pct[i,j] > 50 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/confusion_matrix.png")


def collect_feature_importance(tree):
    """收集各特征的平均信息增益"""
    feature_gains = {}
    def collect_gains(node):
        if node['type'] == 'split':
            fname = node['feature_name']
            if fname not in feature_gains:
                feature_gains[fname] = []
            feature_gains[fname].append(node['info_gain'])
            for branch in node['branches'].values():
                collect_gains(branch)
    collect_gains(tree.tree)

    avg_gains = {k: np.mean(v) for k, v in feature_gains.items()}
    sorted_features = sorted(avg_gains.items(), key=lambda x: x[1], reverse=True)
    return sorted_features


def draw_feature_importance(sorted_features):
    """绘制特征重要性（信息增益排序）"""
    print("绘制: 特征重要性图...")
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [f[0] for f in sorted_features]
    gains = [f[1] for f in sorted_features]
    colors_feat = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), gains, color=colors_feat, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('信息增益 (Information Gain)', fontsize=12)
    ax.set_title('ID3 特征重要性（信息增益排序）', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    for bar, val in zip(bars, gains):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/feature_importance.png")


def draw_tree_structure(tree, stats):
    """绘制树结构示意（前3层）"""
    print("绘制: 树结构示意...")
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')

    lines = [f"ID3 决策树 (深度={stats['depth']}, 节点数={stats['total_nodes']}, 叶子={stats['leaf_nodes']})"]
    lines.append("=" * 80)

    def build_text_limited(node, indent=0, max_depth=3):
        prefix = "  " * indent
        if node['type'] == 'leaf' or indent >= max_depth:
            suffix = f" (n={node['count']})" if node['type'] == 'leaf' else f" (子节点 {len(node['branches'])} 个)"
            return [f"{prefix}└── {node.get('feature_name', '')} {node.get('label', '')}{suffix}"]
        txt = [f"{prefix}├── {node['feature_name']} (IG={node['info_gain']})"]
        for val, branch in node['branches'].items():
            txt.append(f"{prefix}│  ├── ={val}:")
            txt.extend(build_text_limited(branch, indent + 2, max_depth))
        return txt

    lines.extend(build_text_limited(tree.tree, max_depth=3))
    lines.append(f"... (完整树共 {stats['total_nodes']} 节点, {stats['leaf_nodes']} 叶子, 深度 {stats['depth']})")

    ax.text(0.01, 0.99, '\n'.join(lines), fontsize=9, fontfamily='Microsoft YaHei',
            va='top', ha='left', transform=ax.transAxes)
    plt.savefig(os.path.join(PIC_DIR, 'tree_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/tree_structure.png")


def draw_learning_curve(X_train, y_train, X_test, y_test, feature_names):
    """绘制学习曲线（树大小和准确率随训练数据变化）"""
    print("绘制: 学习曲线...")
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tree_sizes, leaf_counts, train_accs, test_accs = [], [], [], []

    for ratio in ratios:
        n_samples = int(len(X_train) * ratio)
        X_sub = X_train[:n_samples]
        y_sub = y_train[:n_samples]
        sub_tree = ID3DecisionTree()
        sub_tree.fit(X_sub, y_sub.tolist(), feature_names)
        stats = sub_tree.get_tree_stats()
        tree_sizes.append(stats['total_nodes'])
        leaf_counts.append(stats['leaf_nodes'])
        train_accs.append(accuracy_score(y_sub, sub_tree.predict(X_sub)))
        test_accs.append(accuracy_score(y_test, sub_tree.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(ratios, tree_sizes, 'bo-', linewidth=2, markersize=6, label='总节点数')
    axes[0].plot(ratios, leaf_counts, 'rs-', linewidth=2, markersize=6, label='叶子节点数')
    axes[0].set_xlabel('训练集比例', fontsize=12)
    axes[0].set_ylabel('节点数', fontsize=12)
    axes[0].set_title('ID3 树大小随训练数据变化', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ratios, train_accs, 'g^-', linewidth=2, markersize=6, label='训练集准确率')
    axes[1].plot(ratios, test_accs, 'md-', linewidth=2, markersize=6, label='测试集准确率')
    axes[1].set_xlabel('训练集比例', fontsize=12)
    axes[1].set_ylabel('准确率', fontsize=12)
    axes[1].set_title('ID3 准确率随训练数据变化', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: pic/learning_curve.png")


# ================================================================
# 保存结果
# ================================================================

def save_results(tree, X_train, X_test, y_test, y_test_pred, cm, pos_order,
                 sorted_features, stats, train_acc, test_acc):
    """保存所有实验结果到 CSV 和 TXT 文件"""
    # 预测结果
    pd.DataFrame({
        '真实位置': [pos_order[i] for i in y_test],
        '预测位置': [pos_order[i] for i in y_test_pred],
        '正确': [y_test[i] == y_test_pred[i] for i in range(len(y_test))]
    }).to_csv(os.path.join(RESULT_DIR, 'prediction_results.csv'), index=False, encoding='utf-8-sig')

    # 混淆矩阵
    pd.DataFrame(cm, index=pos_order, columns=pos_order).to_csv(
        os.path.join(RESULT_DIR, 'confusion_matrix.csv'), encoding='utf-8-sig')

    # 保存树结构 JSON
    with open(os.path.join(RESULT_DIR, 'tree_structure.json'), 'w', encoding='utf-8') as f:
        json.dump(tree.tree, f, indent=2, ensure_ascii=False)

    # 分类规则
    rules = tree.get_rules()
    rules_text = []
    for i, (conditions, label, count) in enumerate(rules):
        rule_str = " AND ".join(conditions)
        rules_text.append(f"Rule{i+1}: IF {rule_str} THEN Pos={pos_order[label]} (samples={count})")
    with open(os.path.join(RESULT_DIR, 'decision_rules.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rules_text))

    # 特征重要性
    pd.DataFrame({
        'feature': [f[0] for f in sorted_features],
        'avg_info_gain': [f[1] for f in sorted_features]
    }).to_csv(os.path.join(RESULT_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')

    # 实验摘要
    summary = pd.DataFrame([{
        'algorithm': 'ID3 Decision Tree',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(FEATURE_LIST),
        'tree_depth': stats['depth'],
        'total_nodes': stats['total_nodes'],
        'leaf_nodes': stats['leaf_nodes'],
        'train_accuracy': round(train_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'overfitting_gap': round(train_acc - test_acc, 4),
        'num_rules': len(rules),
    }])
    summary.to_csv(os.path.join(RESULT_DIR, 'experiment_summary.csv'), index=False, encoding='utf-8-sig')



# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("ID3 决策树实验 — NBA 球员位置预测")
    print("=" * 60)

    # 1. 数据预处理
    df = preprocess_data()

    # 2. 特征离散化
    df_binned = discretize_features(df)

    # 3. 编码
    X, y, feature_names, pos_order = encode_features(df_binned)
    display_names = [col.replace('_bin', '') for col in feature_names]
    print(f"\n特征矩阵: {X.shape}")
    print(f"特征名: {feature_names}")

    # 4. 划分数据集
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. 训练 ID3
    tree, stats = train_id3(X_train, y_train, display_names)

    # 6. 打印树结构和规则
    tree.print_tree()
    tree.print_rules(top_k=15)

    # 7. 评估
    train_acc, test_acc, y_train_pred, y_test_pred, cm = evaluate_id3(
        tree, X_train, y_train, X_test, y_test, pos_order
    )

    # 8. 可视化
    print("\n" + "=" * 60)
    print("6. 可视化")
    print("=" * 60)
    draw_confusion_matrix(cm, pos_order, test_acc)
    sorted_features = collect_feature_importance(tree)
    draw_feature_importance(sorted_features)
    draw_tree_structure(tree, stats)
    draw_learning_curve(X_train, y_train, X_test, y_test, display_names)

    # 9. 保存结果
    print("\n" + "=" * 60)
    print("7. 保存结果")
    print("=" * 60)
    save_results(tree, X_train, X_test, y_test, y_test_pred, cm,
                 pos_order, sorted_features, stats, train_acc, test_acc)

    print("=" * 60)
    print("ID3 决策树实验完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
