import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示（防止乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 加载数据 ==================
# 假设你的数据保存在这个文件中，用你实际的路径替换
df = pd.read_csv('./data/NBA_Season_Stats.csv')

# 查看数据结构
print("数据前5行：")
print(df.head())
print("\n数据列名：", df.columns.tolist())
print("数据维度：", df.shape)

# ================== 2. 数据清洗与特征选择 ==================
# 剔除唯一标识符和非数值列（Player, Pos, Tm 是字符串，不能直接算相关系数）
# Year 虽然是数值，但如果是单一年份则没有分析意义，如果是多年数据可以保留
categorical_cols = ['Player', 'Pos', 'Tm']  # 需要排除的文本列
numeric_df = df.drop(columns=categorical_cols, errors='ignore')

# 确保所有列都是数值类型，如果 Year 被识别为 object，转为 int
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

# 处理缺失值（如果有空值，直接删除该行，或填充中位数）
# 因为篮球数据通常完整，这里采用删除法
numeric_df = numeric_df.dropna()

print(f"清洗后用于分析的数值特征：{numeric_df.columns.tolist()}")
print(f"有效数据行数：{numeric_df.shape[0]}")

# ================== 3. 计算相关性矩阵 ==================
corr_matrix = numeric_df.corr(method='pearson')  # 也可用 'spearman' 处理非线性单调关系

# ================== 4. 可视化：热力图（全貌） ==================
plt.figure(figsize=(18, 15))
# annot=True 显示数值，fmt='.2f' 保留两位小数，cmap选择配色
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('所有数值特征之间的相关性矩阵热力图', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ================== 5. 聚焦目标变量：找出与得分（PTS）最相关的特征 ==================
if 'PTS' in corr_matrix.columns:
    # 取出 PTS 与其他所有特征的相关系数，按绝对值排序
    pts_corr = corr_matrix['PTS'].sort_values(ascending=False)
    print("\n与得分 (PTS) 的相关系数排名（从高到低）：")
    print(pts_corr)

# ================== 6. 进阶可视化：挑选关键特征做散点矩阵 ==================
# 如果特征太多，散点矩阵会很拥挤，这里挑选几个经典的高关联特征
key_features = ['MP', 'FGA', 'FG%', '3P', 'FTA', 'TRB', 'AST', 'PTS']
# 检查这些特征是否都在数据中
existing_features = [col for col in key_features if col in numeric_df.columns]

if len(existing_features) > 1:
    sns.pairplot(numeric_df[existing_features], diag_kind='kde', 
                 plot_kws={'alpha':0.6, 's':30})
    plt.suptitle('关键特征两两散点分布矩阵', y=1.02)
    plt.show()

# ================== 7. (可选) 导出相关性矩阵为 CSV 方便查看 ==================
corr_matrix.to_csv('correlation_matrix.csv')
print("相关性矩阵已导出为 correlation_matrix.csv")