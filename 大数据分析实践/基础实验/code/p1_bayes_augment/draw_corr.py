import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ================== 1. 从CSV导入数据 ==================
# 请将文件路径替换为你实际的CSV文件路径
file_path = 'feature_correlation_matrix.csv'  # 修改为你的文件路径

# 读取CSV文件
# 注意：你的数据中第一列是特征名称，第一行也是特征名称
df = pd.read_csv(file_path, index_col=0)  # 将第一列设为行索引

# 查看数据基本信息
print("数据形状：", df.shape)
print("\n前5行数据预览：")
print(df.head())
print("\n数据列名：")
print(df.columns.tolist())

# ================== 2. 数据清洗与验证 ==================
# 确保所有数据都是数值类型
df = df.apply(pd.to_numeric, errors='coerce')

# 检查是否有缺失值
if df.isnull().any().any():
    print("\n警告：数据中存在缺失值，将用0填充")
    df = df.fillna(0)

# 检查是否为对称矩阵（相关性矩阵应该是对称的）
if not np.allclose(df, df.T, rtol=1e-5):
    print("\n警告：矩阵不是完全对称的，将自动对称化处理")
    df = (df + df.T) / 2

# 检查对角线的值是否接近1
diag_values = np.diag(df)
if not np.allclose(diag_values, 1.0, rtol=1e-5):
    print("\n警告：对角线值不全为1，将进行修正")
    np.fill_diagonal(df.values, 1.0)

print(f"\n清洗后的数据形状：{df.shape}")
print(f"特征列表：{df.index.tolist()}")

# ================== 3. 绘制热力图 ==================
# 设置图形大小（根据特征数量动态调整）
n_features = df.shape[0]
fig_size = max(12, n_features * 0.8)  # 动态调整大小
fig, ax = plt.subplots(figsize=(fig_size, fig_size))

# 绘制热力图
# annot=True 显示数值，fmt='.2f'保留两位小数，cmap选择配色方案
heatmap = sns.heatmap(
    df,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',  # 红蓝配色，红色正相关，蓝色负相关
    center=0,         # 以0为中心
    square=True,      # 方形格子
    linewidths=0.5,   # 格子间距线
    linecolor='white',
    cbar_kws={"shrink": 0.8, "label": "相关系数"},
    ax=ax,
    annot_kws={'size': 8} if n_features > 10 else {'size': 10}  # 特征多时自动缩小字体
)

# 设置标题
plt.title('篮球技术统计指标相关性矩阵热力图', fontsize=16, fontweight='bold', pad=20)

# 调整x轴标签（旋转45度，右对齐）
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# 调整布局防止标签被裁剪
plt.tight_layout()

# 保存图片（可选）
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n热力图已保存为 'correlation_heatmap.png'")

# 显示图形
plt.show()

# ================== 4. 额外：找出最强的正负相关关系 ==================
print("\n" + "="*60)
print("最强正相关关系（Top 10）：")
print("="*60)

# 提取上三角矩阵（避免重复）
mask = np.triu(np.ones(df.shape), k=1).astype(bool)
corr_pairs = df.where(mask).stack().sort_values(ascending=False)

# 显示最强的10个正相关
for i, (pair, value) in enumerate(corr_pairs.head(10).items()):
    if i < 10:
        print(f"{i+1}. {pair[0]} ↔ {pair[1]} : {value:.4f}")

print("\n" + "="*60)
print("最强负相关关系（Top 10）：")
print("="*60)
for i, (pair, value) in enumerate(corr_pairs.tail(10).items()):
    print(f"{i+1}. {pair[0]} ↔ {pair[1]} : {value:.4f}")

# ================== 5. 可选：导出清洗后的数据 ==================
df.to_csv('correlation_matrix_cleaned.csv')
print("\n清洗后的数据已保存为 'correlation_matrix_cleaned.csv'")