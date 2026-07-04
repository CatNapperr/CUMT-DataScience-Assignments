"""
IRIS鸢尾花聚类分析
比较K-means和GMM-EM两种聚类算法的性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score


# ============ K-means聚类类 ============
class KMeans:
    """K-means聚类算法实现"""
    
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = None
        self.labels = None
    
    def fit_predict(self, data):
        """拟合数据并返回聚类标签"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = data.shape[0]
        
        # 随机初始化中心点
        init_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centers = data[init_indices].copy()
        
        for iteration in range(self.max_iter):
            # 分配样本到最近的中心
            distances = np.zeros((n_samples, self.n_clusters))
            for k in range(self.n_clusters):
                distances[:, k] = np.sum((data - self.centers[k]) ** 2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # 更新中心点
            new_centers = np.zeros_like(self.centers)
            for k in range(self.n_clusters):
                cluster_points = data[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = cluster_points.mean(axis=0)
                else:
                    new_centers[k] = self.centers[k]
            
            # 检查收敛
            if np.allclose(self.centers, new_centers):
                break
            
            self.centers = new_centers
        
        self.labels = labels
        return labels


# ============ GMM-EM聚类类 ============
class GMM_EM:
    """混合高斯模型EM算法实现"""
    
    def __init__(self, n_components=3, max_iter=1000, error=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.alpha = None      # 混合权重
        self.mu = None         # 均值
        self.sigma = None      # 协方差
        self.labels = None
    
    def _init_params(self, data):
        """初始化模型参数"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = data.shape
        self.mu = np.random.rand(self.n_components, n_features)
        self.sigma = np.array([np.eye(n_features) / n_features] * self.n_components)
        self.alpha = np.ones(self.n_components) / self.n_components
    
    def _gauss_pdf(self, data, mu, sigma):
        """计算多元高斯概率密度"""
        return multivariate_normal.pdf(data, mean=mu, cov=sigma)
    
    def _e_step(self, data):
        """E步：计算后验概率"""
        n_samples = data.shape[0]
        probs = np.zeros((n_samples, self.n_components))
        
        # 计算条件概率
        for k in range(self.n_components):
            probs[:, k] = self.gauss_pdf(data, self.mu[k], self.sigma[k])
        
        # 计算后验概率
        weighted_probs = self.alpha * probs
        weighted_probs /= weighted_probs.sum(axis=1, keepdims=True)
        
        return weighted_probs
    
    def _m_step(self, data, weighted_probs):
        """M步：更新模型参数"""
        n_samples, n_features = data.shape
        
        for k in range(self.n_components):
            # 计算有效样本数
            Nk = weighted_probs[:, k].sum()
            
            # 更新均值
            self.mu[k] = (weighted_probs[:, k, np.newaxis] * data).sum(axis=0) / Nk
            
            # 更新协方差
            diff = data - self.mu[k]
            self.sigma[k] = (weighted_probs[:, k, np.newaxis, np.newaxis] * 
                            diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0) / Nk
            
            # 更新混合权重
            self.alpha[k] = Nk / n_samples
    
    def fit_predict(self, data):
        """拟合数据并返回聚类标签"""
        # 数据归一化
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        
        self._init_params(data)
        
        for iteration in range(self.max_iter):
            # E步
            weighted_probs = self._e_step(data)
            
            # 检查收敛
            if iteration > 0:
                change = np.linalg.norm(weighted_probs - prev_weighted_probs)
                if change < self.error:
                    break
            
            prev_weighted_probs = weighted_probs
            
            # M步
            self._m_step(data, weighted_probs)
        
        self.labels = weighted_probs.argmax(axis=1)
        return self.labels
    
    def gauss_pdf(self, data, mu, sigma):
        """计算多元高斯概率密度（供_e_step调用）"""
        return multivariate_normal.pdf(data, mean=mu, cov=sigma)


# ============ 数据加载与预处理 ============
def load_and_prepare_data(filepath):
    """加载IRIS数据集并提取用于聚类的特征"""
    # 加载数据
    iris_data = pd.read_csv(
        filepath, 
        header=None, 
        names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    )
    
    # 编码标签
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(iris_data['class'].values)
    
    # 提取聚类特征（sepal length和petal length）
    X = iris_data[['sepal length', 'petal length']].values
    
    return X, true_labels, iris_data


# ============ 标签匹配与评估 ============
def find_best_label_mapping(pred_labels, true_labels):
    """找到预测标签和真实标签的最优匹配"""
    from itertools import permutations
    
    n_clusters = len(np.unique(pred_labels))
    best_accuracy = 0
    
    # 尝试所有可能的标签排列
    for perm in permutations(range(n_clusters)):
        # 创建标签映射
        mapped_labels = pred_labels.copy()
        for i, j in enumerate(perm):
            mapped_labels[pred_labels == i] = j + n_clusters
        mapped_labels -= n_clusters
        
        # 计算准确率
        accuracy = np.mean(mapped_labels == true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = perm
    
    return best_accuracy, best_mapping


def calculate_metrics(pred_labels, true_labels):
    """计算聚类评估指标"""
    # 准确率（需要标签对齐）
    accuracy, _ = find_best_label_mapping(pred_labels, true_labels)
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    return {'accuracy': accuracy, 'ari': ari}


# ============ 可视化 ============
def plot_clustering_results(X, pred_labels, centers, true_labels, title):
    """绘制聚类结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 聚类结果
    axes[0].scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis', alpha=0.6)
    axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3)
    axes[0].set_xlabel('Sepal Length')
    axes[0].set_ylabel('Petal Length')
    axes[0].set_title(f'{title} - 聚类结果')
    axes[0].grid(True, alpha=0.3)
    
    # 真实标签
    axes[1].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.6)
    axes[1].set_xlabel('Sepal Length')
    axes[1].set_ylabel('Petal Length')
    axes[1].set_title('真实分类')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============ 主程序 ============
def main():
    print("=" * 60)
    print("IRIS鸢尾花聚类分析")
    print("=" * 60)
    
    # 1. 加载和准备数据
    print("\n[1] 加载数据...")
    X, true_labels, iris_data = load_and_prepare_data("Iris/iris.data")
    
    print(f"数据集大小: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")
    print(f"聚类数: {len(np.unique(true_labels))}")
    print(f"类别分布: {np.bincount(true_labels)}")
    
    # 2. K-means聚类
    print("\n[2] 执行K-means聚类...")
    kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_metrics = calculate_metrics(kmeans_labels, true_labels)
    
    print(f"K-means 准确率: {kmeans_metrics['accuracy']:.4f}")
    print(f"K-means ARI指数: {kmeans_metrics['ari']:.4f}")
    
    plot_clustering_results(X, kmeans_labels, kmeans.centers, true_labels, "K-means")
    
    # 3. GMM-EM聚类
    print("\n[3] 执行GMM-EM聚类...")
    gmm = GMM_EM(n_components=3, max_iter=1000, error=1e-6, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    gmm_metrics = calculate_metrics(gmm_labels, true_labels)
    
    print(f"GMM-EM 准确率: {gmm_metrics['accuracy']:.4f}")
    print(f"GMM-EM ARI指数: {gmm_metrics['ari']:.4f}")
    
    # 可视化GMM聚类中心（反归一化）
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    gmm_centers_normalized = gmm.mu
    gmm_centers = scaler.inverse_transform(gmm_centers_normalized)
    
    plot_clustering_results(X, gmm_labels, gmm_centers, true_labels, "GMM-EM")
    
    # 4. 算法对比
    print("\n" + "=" * 60)
    print("算法对比总结")
    print("=" * 60)
    print(f"{'指标':<15} {'K-means':<15} {'GMM-EM':<15}")
    print("-" * 45)
    print(f"{'准确率':<15} {kmeans_metrics['accuracy']:<15.4f} {gmm_metrics['accuracy']:<15.4f}")
    print(f"{'ARI指数':<15} {kmeans_metrics['ari']:<15.4f} {gmm_metrics['ari']:<15.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
