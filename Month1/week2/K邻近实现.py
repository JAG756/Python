# ======================
# Day10-1：K近邻 KNN 分类
# 直接运行！
# ======================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 标签（花的种类）

# 2. 划分训练集 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建 K近邻 模型
knn = KNeighborsClassifier(n_neighbors=3)  # k=3

# 4. 训练
knn.fit(X_train, y_train)

# 5. 预测
y_pred = knn.predict(X_test)

# 6. 评分
print("===== K近邻 结果 =====")
print(f"预测准确率: {accuracy_score(y_test, y_pred):.2f}")
print(f"模型分数: {knn.score(X_test, y_test):.2f}")