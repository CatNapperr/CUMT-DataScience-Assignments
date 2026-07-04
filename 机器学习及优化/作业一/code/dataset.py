"""
房屋价格预测模型 - 线性回归
使用梯度下降法和正规方程进行模型训练
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==================== 数据加载和预处理 ====================
# 读取CSV数据文件
data = pd.read_csv('housing.csv')
print("原始数据:")
print(data)

# 检查数据结构和基本信息
print("\n数据信息:")
print(data.info())
print("\noceam_proximity 字段分布:")
print(data['ocean_proximity'].value_counts())

# 将分类特征 ocean_proximity 转换为独热编码（one-hot encoding）
data_1 = pd.get_dummies(data, columns=['ocean_proximity'], prefix='ocean', dtype=int)

# 删除包含缺失值的行（移除 total_bedrooms 为空的数据）
data_1 = data_1.dropna(axis=0, subset=["total_bedrooms"])

print("\n处理后的数据:")
print(data_1)

# ==================== 特征和标签分离 ====================
# 将目标变量 median_house_value 分离出来
tmp = data_1.drop("median_house_value", axis=1)
y = data_1['median_house_value']
# 重新拼接数据，目标变量放在最后一列
data_2 = pd.concat([tmp, y], axis=1)
print("\n最终数据集形状:", data_2.shape)
print(data_2)

# ==================== 训练集和测试集划分 ====================
# 按照7:3的比例划分训练集和测试集，使用随机种子确保可重复性
train_data, test_data = train_test_split(data_2, test_size=0.3, random_state=42)

print("\n训练集:")
print(train_data)
print("\n测试集:")
print(test_data)

# ==================== 构建特征矩阵和标签矩阵 ====================
# 获取数据集的列数，用于分割特征和标签
n = train_data.shape[1]

# 从训练集中分离特征 X 和标签 y
# X: 包含所有特征列（除了最后一列的目标变量）
# y: 目标变量列（最后一列）
X = train_data.iloc[:, 0:n - 1]
y = train_data.iloc[:, n - 1:n]

# 从测试集中分离特征和标签
X_test = test_data.iloc[:, 0:n - 1]
y_test = test_data.iloc[:, n - 1:n]

print("\nX 数据（特征矩阵）:")
print(X)
print("\ny 数据（标签向量）:")
print(y)

# ==================== 数据类型转换 ====================
# 将 DataFrame 转换为 numpy 矩阵，便于后续的矩阵运算
X = np.matrix(X.values)
y = np.matrix(y.values)
X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)

print("\nX 矩阵（numpy格式）:")
print(X)
print("\ny 矩阵（numpy格式）:")
print(y)
print("\nX 的形状: {}".format(X.shape))
print("y 的形状: {}".format(y.shape))

# ==================== 特征标准化（归一化） ====================
def norm(X):
    """
    对特征矩阵进行标准化处理
    
    参数:
        X (np.matrix): 输入的特征矩阵
        
    返回:
        X (np.matrix): 标准化后的特征矩阵
        
    标准化公式: X_norm = (X - mean) / std
    """
    # 计算每一列（每个特征）的标准差
    sigma = np.std(X, axis=0)
    # 计算每一列的均值
    miu = np.mean(X, axis=0)
    # 应用标准化公式
    X = (X - miu) / sigma
    return X


# 对训练集和测试集进行特征标准化
X = norm(X)
X_test = norm(X_test)

# ==================== 添加偏置项 ====================
# 在特征矩阵的第一列添加全1，对应参数 theta0（偏置项）
X = np.c_[np.ones(y.size), X]
X_test = np.c_[np.ones(y_test.size), X_test]

print("\n标准化后的 X 矩阵:")
print(X)
print("\n标准化后的 X_test 矩阵:")
print(X_test)
print("\nX 的形状: {}".format(X.shape))
print("X_test 的形状: {}".format(X_test.shape))

# ==================== 梯度下降法参数初始化 ====================
# 初始化参数向量 theta（包括偏置项），维度为 (特征数+1, 1)
theta = np.matrix(np.zeros((X.shape[1], 1)))
theta_2 = theta

# 设置超参数
num_iteration = 20000  # 迭代次数
alpha = 0.01           # 学习速率

# 初始化代价函数值列表，用于记录每次迭代的损失
J = np.zeros(num_iteration)

# ==================== 代价函数定义 ====================
def cost(theta, X=X, y=y, n=y.size):
    """
    计算均方误差代价函数
    
    参数:
        theta (np.matrix): 参数向量
        X (np.matrix): 特征矩阵
        y (np.matrix): 目标变量
        n (int): 样本数
        
    返回:
        cost (float): 代价函数值
        
    公式: J(theta) = 1/(2*m) * sum((h(x) - y)^2)
    """
    # 计算预测值
    h_x = X @ theta
    # 计算误差的平方和
    inner = np.sum(np.power(h_x - y, 2))
    # 返回平均代价
    return inner / (2 * n)


# ==================== 梯度下降法实现 ====================
def gradient_descend(theta, alpha):
    """
    使用梯度下降法优化参数
    
    参数:
        theta (np.matrix): 参数向量
        alpha (float): 学习速率
        
    返回:
        theta (np.matrix): 优化后的参数向量
        
    更新规则: theta = theta - (alpha/m) * X^T * (X*theta - y)
    """
    for i in range(num_iteration):
        # 记录当前迭代的代价函数值
        J[i] = cost(theta, X)
        # 梯度下降更新参数
        theta = theta - (alpha / y.size) * (X.T @ (X @ theta - y))
    return theta


# ==================== 梯度下降法求解 ====================
theta = gradient_descend(theta, alpha)

print("\n梯度下降法结果:")
print("学习速率: {}".format(alpha))
print("求得的 θ 参数:\n{}".format(theta))

# ==================== 可视化代价函数变化 ====================
# 绘制代价函数随迭代次数的变化曲线
plt.figure(0)
plt.plot(J)
plt.xlabel('迭代次数 (Steps)')
plt.ylabel('代价函数值 (Loss)')
plt.title('线性回归 - 梯度下降法学习曲线')
plt.grid(True)
plt.show()

# ==================== 模型评估 - R² 指标 ====================
def R_2(X_test, y_test, theta):
    """
    计算决定系数 R² 用于模型评估
    
    参数:
        X_test (np.matrix): 测试集特征矩阵
        y_test (np.matrix): 测试集标签
        theta (np.matrix): 模型参数
        
    返回:
        r_2 (float): R² 分数 (0-1，越接近1越好)
        
    公式: R² = 1 - SSE/SST = 1 - SSE/(SSR + SSE)
        其中 SSE = sum((y_pred - y_test)^2)  # 残差平方和
             SSR = sum((y_pred - y_mean)^2)  # 回归平方和
             SST = sum((y_test - y_mean)^2)  # 总平方和
    """
    # 计算预测值
    y_pred = X_test * theta
    # 计算标签的均值
    mu = np.mean(y_test, axis=0)
    # 计算残差平方和
    SSE = np.sum(np.power(y_test - y_pred, 2))
    # 计算回归平方和
    SSR = np.sum(np.power(y_pred - mu, 2))
    # 计算总平方和
    SST = SSR + SSE
    # 计算 R²
    r_2 = 1 - SSE / SST
    return r_2


# 计算梯度下降法的性能
print("\n==================== 梯度下降法性能评估 ====================")
print("测试集上的 R²: {:.6f}".format(R_2(X_test, y_test, theta)))
print("训练集上的 R²: {:.6f}".format(R_2(X, y, theta)))

# ==================== 正规方程法求解 ====================
def NormalEquation(theta_2):
    """
    使用正规方程法求解参数
    
    参数:
        theta_2 (np.matrix): 初始参数向量
        
    返回:
        theta_2 (np.matrix): 最优参数向量
        
    公式: theta = (X^T * X)^(-1) * X^T * y
    """
    # 直接通过正规方程计算最优参数（无需迭代）
    theta_2 = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_2


# 求解参数
theta_2 = NormalEquation(theta_2)

print("\n==================== 正规方程法结果 ====================")
print("求得的 θ 参数:\n{}".format(theta_2))
print("\n测试集上的 R²: {:.6f}".format(R_2(X_test, y_test, theta_2)))

# ==================== 特征相关性分析 ====================
# 计算数据的相关系数矩阵
temp = data_1.copy()
corr = temp.corr()
# 获取与目标变量的相关系数并排序
score = corr['median_house_value'].sort_values()

print("\n==================== 特征与目标变量的相关系数 ====================")
print("（从弱到强排列）:")
print(score)